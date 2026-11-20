"""Download a Planetoid dataset and save it as a PyG graph with node masks.

    python test/planetoid_split_save.py                 # -> data/Cora.pt
    python test/planetoid_split_save.py --split random --split-ratio 0.7 --normalize

The graph is saved *intact* -- every node, every edge -- and the train/val/test
split is expressed as boolean node masks, which is the convention PyG expects
for node-level tasks.

The previous version removed nodes while keeping the original ``edge_index``:

    train_data = Data(x=data.x[train_idx], edge_index=data.edge_index, ...)

so edge endpoints pointed past the end of ``x``. That is the bug that produced

    IndexError: index 2707 is out of bounds for dimension 0 with size 2166

on CPU and, for the same kernel on GPU,
``CUDA error: device-side assert triggered``. If an induced subgraph is ever
really needed, use ``torch_geometric.utils.subgraph``, which relabels the edges
and keeps the graph consistent.

Downloading happens here, on a login node: JSC production partitions
(``dc-gpu``) have no internet access, so the compute node can only read a graph
that already exists on disk.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.preprocess import graph_summary, normalize_node_features, validate_graph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a Planetoid dataset as a masked PyG graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="Cora", choices=["Cora", "CiteSeer", "PubMed"], help="Planetoid dataset"
    )
    parser.add_argument("--out-dir", default="data", help="output directory (also the download root)")
    parser.add_argument(
        "--split",
        default="official",
        choices=["official", "random"],
        help="'official' keeps the Planetoid masks, 'random' draws a seeded random split",
    )
    parser.add_argument("--split-ratio", type=float, default=0.8, help="train fraction (random split)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="val fraction (random split)")
    parser.add_argument("--normalize", action="store_true", help="standardise node features")
    parser.add_argument("--seed", type=int, default=42, help="seed for the random split")
    return parser.parse_args()


def random_masks(num_nodes: int, split_ratio: float, val_ratio: float, seed: int) -> dict:
    """Boolean train/val/test masks over a seeded permutation of all nodes."""
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("--split-ratio must be in (0, 1).")
    if not 0.0 <= val_ratio < 1.0 - split_ratio:
        raise ValueError("--val-ratio must leave room for a test split.")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)
    num_train = int(round(num_nodes * split_ratio))
    num_val = int(round(num_nodes * val_ratio))
    num_test = num_nodes - num_train - num_val
    if min(num_train, num_val, num_test) < 1:
        raise ValueError("Split leaves at least one node empty; adjust the ratios.")

    masks = {}
    for name, count, offset in (
        ("train", num_train, 0),
        ("val", num_val, num_train),
        ("test", num_test, num_train + num_val),
    ):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[perm[offset : offset + count]] = True
        masks[f"{name}_mask"] = mask
    return masks


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Planetoid(root=str(out_dir / "Planetoid"), name=args.dataset)
    data = dataset[0]

    if args.normalize:
        data = normalize_node_features(data)

    if args.split == "random":
        for name, mask in random_masks(
            int(data.num_nodes), args.split_ratio, args.val_ratio, args.seed
        ).items():
            setattr(data, name, mask)
    else:
        missing = [
            split for split in ("train", "val", "test") if getattr(data, f"{split}_mask", None) is None
        ]
        if missing:
            raise RuntimeError(
                f"{args.dataset} has no {missing} mask; use --split random instead."
            )

    validate_graph(data, require_masks=True)

    out_path = out_dir / f"{args.dataset}.pt"
    torch.save(data, out_path)

    summary = graph_summary(data)
    print(f"Saved {args.dataset} graph -> {out_path}")
    print(f"  nodes={summary['num_nodes']} edges={summary['num_edges']} "
          f"features={summary['num_node_features']} classes={summary.get('num_classes', '-')}")
    print(f"  train={summary['train_nodes']} val={summary['val_nodes']} test={summary['test_nodes']} "
          f"| edge_index_max={summary['edge_index_max']} (< {summary['num_nodes']})")
    print(f"  checks: consistent graph OK")
    print(
        "  If `model.input_dim`/`model.output_dim` are set explicitly in config.yaml they must "
        f"match features={summary['num_node_features']}, output={summary.get('num_classes', 'target_dim')}; "
        "leave them null to infer."
    )


if __name__ == "__main__":
    main()
