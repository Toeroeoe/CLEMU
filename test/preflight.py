"""Pre-flight check for CLEMU, intended to run as the first srun step of a job.

    python test/preflight.py                  # auto-detect device, include the data check
    python test/preflight.py --device cuda    # insist on the GPU path
    python test/preflight.py --skip-data      # environment only

Why this exists: the device-specific failures cannot be reproduced off-cluster.
This machine (and any login node) has no GPU, so NCCL setup, device placement and
device-side assertions can only be observed on the compute node -- and they
surface there as opaque messages like

    RuntimeError: CUDA error: device-side assert triggered
    Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Rather than debug that from an sbatch log, the job now runs this first: it prints
the exact stack (torch build, CUDA, driver, arch list, NCCL), moves a real tensor
through the two kernels that fail most often (index-select and scatter-add), and
runs one forward/backward of the actual model on the actual graph. Every check is
independent, so a failure names the thing that is broken instead of a stack trace.

Exit code 0 only if every check passes; "PREFLIGHT OK" is the last line, which
makes it greppable from the job log.
"""

import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

# PyG still calls torch.jit.script on import, which is deprecated in torch >= 2.6
# and would otherwise interleave into the report.
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS = []
REQUIRE_CUDA = False


def record(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def check_environment() -> None:
    print("Environment")
    rocm = getattr(torch.version, "hip", None)
    record("python", True, f"{sys.version.split()[0]} ({sys.executable})")
    record("torch", True, torch.__version__)
    record("torch cuda build", True, torch.version.cuda or f"none (rocm={rocm})" if not rocm else rocm)
    # A CPU-only host is a valid environment (login node, laptop); it is only a
    # failure when the GPU path was explicitly requested, i.e. on a compute node.
    has_gpu = torch.cuda.device_count() > 0
    record(
        "device count",
        has_gpu or not REQUIRE_CUDA,
        f"{torch.cuda.device_count()} visible GPU(s)"
        f"{' | CUDA_VISIBLE_DEVICES=' + os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else ' | CUDA_VISIBLE_DEVICES unset'}"
        + ("" if has_gpu else " -- CPU-only run" if not REQUIRE_CUDA else " -- but a GPU was requested"),
    )
    if torch.cuda.is_available():
        arch_list = torch.cuda.get_arch_list()
        capability = torch.cuda.get_device_capability(0)
        record(
            "device 0",
            True,
            f"{torch.cuda.get_device_name(0)} sm_{capability[0]}{capability[1]}",
        )
        # "CUDA error: no kernel image is available" is an arch mismatch, not a bug in the model.
        record(
            "arch support",
            f"sm_{capability[0]}{capability[1]}" in arch_list,
            f"build supports {arch_list}",
        )
    record("nccl built", torch.distributed.is_nccl_available(), "")
    record("gloo built", torch.distributed.is_gloo_available(), "")
    dist_env = {
        key: os.environ[key] for key in ("WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT") if key in os.environ
    }
    print(f"  [info] launcher env: {dist_env or 'not set (plain python, no DDP)'}")


def check_device_kernels(device: torch.device) -> None:
    """Exercise index-select and scatter-add, the two kernels behind a device-side assert."""
    print(f"Device kernels on {device}")
    torch.manual_seed(0)
    try:
        x = torch.randn(64, 8, device=device)
        index = torch.randint(0, 64, (256,), device=device)
        gathered = x.index_select(0, index)
        # Same shape mismatch that made a stale edge_index assert on the GPU rather
        # than raise IndexError: src ids reach past dim_size.
        scattered = torch.zeros(64, 8, device=device).scatter_add_(
            0, index.unsqueeze(-1).expand(-1, 8), gathered
        )
        torch.cuda.synchronize() if device.type == "cuda" else None
        record("index_select + scatter_add", bool(torch.isfinite(scattered).all()), "fp32, 64 nodes / 256 edges")
    except RuntimeError as error:
        record("index_select + scatter_add", False, str(error).splitlines()[0])
        return

    try:
        model = torch.nn.Linear(8, 4).to(device)
        loss = model(x).sum()
        loss.backward()
        torch.cuda.synchronize() if device.type == "cuda" else None
        record("autograd forward/backward", model.weight.grad is not None, "Linear(8,4)")
    except RuntimeError as error:
        record("autograd forward/backward", False, str(error).splitlines()[0])

    # Out-of-range index: on CPU this must raise IndexError, on GPU a device-side
    # assert. Both are caught here so the user sees the message, not a hang.
    try:
        bad = torch.zeros(16, 2, device=device)
        bad.index_select(0, torch.tensor([99], device=device, dtype=torch.long))
        torch.cuda.synchronize() if device.type == "cuda" else None
        record("out-of-range index rejected", False, "no error raised -- check is ineffective")
    except (RuntimeError, IndexError) as error:
        record(
            "out-of-range index rejected",
            True,
            str(error).splitlines()[0][:110],
        )


def check_data(device: torch.device, config_path: str) -> None:
    """Load the configured graph and run one forward/backward of the real model."""
    print("Data and model")
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config.get("data") or {}
    data_path = data_cfg.get("path") or config["train"].get("train_data_path")

    if not Path(data_path).exists():
        record("dataset file", False, f"{data_path} not found (run test/planetoid_split_save.py)")
        return
    record("dataset file", True, data_path)

    try:
        from data.preprocess import graph_summary, load_pyg_data

        data = load_pyg_data(data_path, weights_only=False, require_masks=False)
        record("graph consistency", True, str(graph_summary(data)))
    except Exception as error:  # noqa: BLE001 - report, do not hide
        record("graph consistency", False, str(error).splitlines()[0])
        return

    try:
        from models.gnn import GNN
        from utils.engine import run_epoch
        from utils.helpers import get_loss_function, prepare_targets, resolve_dimensions

        task = config["train"].get("task", "classification")
        input_dim, output_dim = resolve_dimensions(data, config["model"], task)
        model = GNN(
            input_dim,
            output_dim,
            int(config["model"]["hidden_dim"]),
            int(config["model"]["num_layers"]),
            aggregation=config["model"].get("aggregation", "mean"),
            dropout=0.0,
            layer_type=config["model"].get("layer_type", "GCNConv"),
            activation=config["model"].get("activation", "relu"),
        ).to(device)

        batch = data.to(device)
        logits = model(batch.x, batch.edge_index)
        record("forward pass", True, f"x{tuple(batch.x.shape)} -> out{tuple(logits.shape)} on {device}")

        mask = getattr(batch, "train_mask", None)
        out_sel = logits[mask] if mask is not None else logits
        y_sel = batch.y[mask] if mask is not None else batch.y
        targets = prepare_targets(y_sel, out_sel, task)
        loss = get_loss_function(config["train"]["loss_function"])(out_sel, targets)
        record("loss", True, f"{type(loss).__name__} = {float(loss.detach()):.4f} on {int(targets.size(0))} nodes")
        loss.backward()
        grads = [p.grad is not None for p in model.parameters()]
        record("backward pass", all(grads), f"{sum(grads)}/{len(grads)} parameter tensors have gradients")
    except Exception as error:  # noqa: BLE001
        record("model forward/backward", False, str(error).splitlines()[0])
        traceback.print_exc()


def main() -> int:
    global REQUIRE_CUDA
    parser = argparse.ArgumentParser(
        description="CLEMU pre-flight check.", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default=None, help="'cpu', 'cuda' or auto (default)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-data", action="store_true", help="environment checks only")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
        REQUIRE_CUDA = device.type == "cuda"
    elif torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    print(f"CLEMU preflight | pid {os.getpid()} | host {os.uname().nodename}")
    check_environment()
    check_device_kernels(device)
    if not args.skip_data:
        check_data(device, args.config)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print()
    if failed:
        print(f"PREFLIGHT FAILED: {len(failed)} check(s): {', '.join(failed)}")
        return 1
    print(f"PREFLIGHT OK ({len(RESULTS)} checks, device {device})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
