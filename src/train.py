"""Train the CLEMU GNN on a prepared PyG graph.

Two launch modes, same file:

    python src/train.py                          # CPU or a single GPU, no DDP
    torchrun --nproc-per-node=N src/train.py     # one process per GPU (DDP)

DDP is only initialised when ``WORLD_SIZE > 1``, so the script can always be
smoke-tested with plain ``python``. The previous version called
``init_process_group("nccl")`` unconditionally in ``main()`` and read
``os.environ["LOCAL_RANK"]`` without a default, so it could not run outside
``torchrun`` at all -- and it initialised NCCL before selecting a device.

Run it from the repository root: ``config.yaml`` and the paths inside it are
resolved relative to the working directory.
"""

import argparse
import copy
import json
import os

import torch
import yaml
from torch_geometric.loader import DataLoader

from data.preprocess import graph_summary, load_pyg_data
from models.gnn import GNN
from utils.engine import run_epoch
from utils.helpers import (
    count_parameters,
    describe_device,
    ensure_parent_dir,
    get_loss_function,
    get_optimizer,
    resolve_dimensions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CLEMU GNN.", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="override train.num_epochs")
    parser.add_argument("--device", default=None, help="'cpu' or 'cuda'; default: auto-detect")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config:
        raise RuntimeError(f"Config file {path!r} is empty or invalid.")
    return config


def setup_distributed(prefer_cuda: bool = True):
    """Return ``(device, rank, world_size, is_ddp)``.

    Under ``torchrun`` the launcher exports ``WORLD_SIZE``/``LOCAL_RANK``. With a
    single process we deliberately skip ``init_process_group`` and the DDP
    wrapper: one rank cannot benefit from DDP, and NCCL/rendezvous setup is the
    single most common source of "CUDA error" reports on a one-GPU job.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = prefer_cuda and torch.cuda.is_available()

    if world_size > 1:
        if not use_cuda:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} but CUDA is not available. "
                "DDP on JSC GPU nodes needs nccl; check `nvidia-smi` and the module stack."
            )
        torch.cuda.set_device(local_rank)  # select the device *before* NCCL init
        torch.distributed.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), torch.distributed.get_rank(), world_size, True

    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    return device, 0, 1, False


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_cfg = config["train"]
    model_cfg = config["model"]
    data_cfg = config.get("data") or {}

    task = str(train_cfg.get("task", "classification")).lower()
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    prefer_cuda = str(args.device).lower() != "cpu"
    device, rank, world_size, is_ddp = setup_distributed(prefer_cuda)
    is_main = rank == 0

    def log(*values) -> None:
        if is_main:
            print(*values, flush=True)

    # ------------------------------------------------------------------ data
    # `data.path` is the current key; `train.train_data_path` keeps old configs working.
    data_path = data_cfg.get("path") or train_cfg.get("train_data_path")
    if not data_path:
        raise KeyError("No dataset path: set `data.path` in the config.")

    data = load_pyg_data(data_path, weights_only=False, require_masks=False)
    summary = graph_summary(data)
    if "train_nodes" not in summary:
        log(
            "WARNING: graph carries no train_mask/val_mask; training on all nodes. "
            "Run test/planetoid_split_save.py to produce a masked graph."
        )

    batch_size = int(train_cfg.get("batch_size", 1))
    if batch_size > 1:
        log(
            f"NOTE: batch_size={batch_size} but the dataset holds a single graph; "
            "each epoch is one full-graph batch (node-level task)."
        )

    loader = DataLoader(
        [data],
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    # ----------------------------------------------------------------- model
    input_dim, output_dim = resolve_dimensions(data, model_cfg, task)
    model = GNN(
        input_dim,
        output_dim,
        int(model_cfg["hidden_dim"]),
        int(model_cfg["num_layers"]),
        aggregation=model_cfg.get("aggregation", "mean"),
        dropout=float(model_cfg.get("dropout", 0.0)),
        layer_type=model_cfg.get("layer_type", "GCNConv"),
        activation=model_cfg.get("activation", "relu"),
    ).to(device)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    num_epochs = int(args.epochs if args.epochs is not None else train_cfg["num_epochs"])
    loss_function = get_loss_function(train_cfg["loss_function"])
    optimizer_cfg = dict(train_cfg["optimizer"])
    optimizer = get_optimizer(
        optimizer_cfg.pop("type"),
        model.parameters(),
        lr=float(optimizer_cfg.pop("learning_rate")),
        **optimizer_cfg,
    )

    log(f"config            : {args.config}")
    log(f"device            : {describe_device(device)} | world_size={world_size} | ddp={is_ddp}")
    log(f"graph             : {summary}")
    log(f"task              : {task}")
    log(f"dims              : input={input_dim} output={output_dim}")
    log(f"parameters        : {count_parameters(model):,}")
    log(f"epochs            : {num_epochs}")

    # ---------------------------------------------------------------- train
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
    eval_every = max(int(train_cfg.get("eval_every", 1)), 1)
    save_path = ensure_parent_dir(train_cfg["model_save_path"])
    best_val = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(
            model, loader, loss_function, task, device, optimizer=optimizer, split="train"
        )
        message = f"Epoch [{epoch:>3}/{num_epochs}] train loss={train_metrics['loss']:.4f}"
        if task == "classification":
            message += f" acc={train_metrics['accuracy']:.4f}"

        if epoch % eval_every == 0 or epoch == num_epochs:
            val_metrics = run_epoch(
                model, loader, loss_function, task, device, optimizer=None, split="val"
            )
            message += f" | val loss={val_metrics['loss']:.4f}"
            if task == "classification":
                message += f" val acc={val_metrics['accuracy']:.4f}"
            history["epoch"].append(epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics.get("accuracy"))

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_epoch = epoch
                core = model.module if hasattr(model, "module") else model
                best_state = copy.deepcopy(core.state_dict())
                if is_main:
                    torch.save(
                        {
                            "state_dict": best_state,
                            "model_cfg": {
                                "input_dim": input_dim,
                                "output_dim": output_dim,
                                "hidden_dim": int(model_cfg["hidden_dim"]),
                                "num_layers": int(model_cfg["num_layers"]),
                                "aggregation": model_cfg.get("aggregation", "mean"),
                                "dropout": float(model_cfg.get("dropout", 0.0)),
                                "layer_type": model_cfg.get("layer_type", "GCNConv"),
                                "activation": model_cfg.get("activation", "relu"),
                            },
                            "task": task,
                            "epoch": epoch,
                            "val_loss": val_metrics["loss"],
                            "config_path": os.path.abspath(args.config),
                        },
                        save_path,
                    )
                    message += "  <- best, saved"
        log(message)

    # The selection criterion is val loss, so the checkpoint on disk holds the
    # best-validation epoch, not the last one. Restore those weights before
    # anything downstream uses the in-memory model: here it matters, the best
    # epoch was 10 while epoch 200 had collapsed to train loss ~0 and val loss
    # ~0.87 (heavily overfitted).
    if best_state is not None:
        core = model.module if hasattr(model, "module") else model
        core.load_state_dict(best_state)

    if is_main:
        history_path = os.path.join(os.path.dirname(os.path.abspath(save_path)), "history.json")
        with open(history_path, "w") as handle:
            json.dump(
                {
                    "config": args.config,
                    "task": task,
                    "best_epoch": best_epoch,
                    "best_val_loss": None if best_epoch < 0 else best_val,
                    "history": history,
                },
                handle,
                indent=2,
            )
        print(f"Best checkpoint : {save_path} (epoch {best_epoch}, val loss {best_val:.4f})")
        print(f"History         : {history_path}")

    if is_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
