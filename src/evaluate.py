"""Evaluate a saved CLEMU checkpoint on a split of the prepared graph.

The checkpoint written by ``src/train.py`` carries its own model config, so the
model is rebuilt from the file rather than from whatever ``config.yaml`` happens
to say now -- that mismatch is exactly how a checkpoint ends up "not loading"
after an innocent-looking config edit.

    python src/evaluate.py                       # test split, default checkpoint
    python src/evaluate.py --split all           # train/val/test in one go
    python src/evaluate.py --checkpoint out/GNN_Cora.pt --plot out/history.json
"""

import argparse
import json
import os

import torch
import yaml

from data.preprocess import load_pyg_data
from models.gnn import GNN
from utils.engine import run_epoch
from utils.helpers import describe_device, get_device, get_loss_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CLEMU GNN checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="path to the YAML config")
    parser.add_argument("--checkpoint", default=None, help="override evaluate.model_load_path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--plot", default=None, help="path to a history.json to plot")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config:
        raise RuntimeError(f"Config file {path!r} is empty or invalid.")
    return config


def build_model(checkpoint: dict, device: torch.device) -> GNN:
    cfg = checkpoint["model_cfg"]
    model = GNN(
        cfg["input_dim"],
        cfg["output_dim"],
        cfg["hidden_dim"],
        cfg["num_layers"],
        aggregation=cfg.get("aggregation", "mean"),
        dropout=cfg.get("dropout", 0.0),
        layer_type=cfg.get("layer_type", "GCNConv"),
        activation=cfg.get("activation", "relu"),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def plot_history(path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display on compute nodes
    import matplotlib.pyplot as plt

    with open(path, "r") as handle:
        history = json.load(handle)["history"]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(history["epoch"], history["train_loss"], label="train loss")
    axis.plot(history["epoch"], history["val_loss"], label="val loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()
    out = os.path.splitext(path)[0] + ".png"
    figure.tight_layout()
    figure.savefig(out, dpi=150)
    print(f"Wrote {out}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_cfg = config.get("evaluate") or {}
    train_cfg = config["train"]
    data_cfg = config.get("data") or {}

    checkpoint_path = args.checkpoint or eval_cfg.get("model_load_path") or train_cfg["model_save_path"]
    data_path = data_cfg.get("path") or train_cfg.get("train_data_path")

    if not os.path.exists(checkpoint_path):
        raise SystemExit(
            f"checkpoint not found: {checkpoint_path!r}\n"
            f"  looked at: --checkpoint, evaluate.model_load_path, train.model_save_path\n"
            f"  train first:  python src/train.py"
        )

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(
            f"{checkpoint_path} is not a CLEMU checkpoint (expected a dict with 'state_dict')."
        )

    task = checkpoint["task"]
    model = build_model(checkpoint, device)
    loss_function = get_loss_function(eval_cfg.get("loss_function") or train_cfg["loss_function"])

    data = load_pyg_data(data_path, weights_only=False, require_masks=False)
    loader = type("Loader", (), {"__iter__": lambda self: iter([data]), "__len__": lambda self: 1})()

    print(f"checkpoint : {checkpoint_path} (epoch {checkpoint.get('epoch')}, device {describe_device(device)})")
    print(f"graph      : {data_path}")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        mask = getattr(data, f"{split}_mask", None)
        if mask is None:
            print(f"{split:>5}: no {split}_mask in the graph -- skipped")
            continue
        metrics = run_epoch(model, loader, loss_function, task, device, optimizer=None, split=split)
        line = f"{split}: loss={metrics['loss']:.4f} nodes={metrics['nodes']}"
        if "accuracy" in metrics:
            line += f" acc={metrics['accuracy']:.4f}"
        print(line)

    if args.plot:
        plot_history(args.plot)


if __name__ == "__main__":
    main()
