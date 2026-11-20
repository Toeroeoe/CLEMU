"""Loading, integrity checks and normalisation for PyG graphs."""

from typing import Any, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

SPLITS = ("train", "val", "test")


def load_data(file_path: str) -> pd.DataFrame:
    """Load tabular data (legacy CSV path, kept for the tabular baseline)."""
    return pd.read_csv(file_path)


def preprocess_data(data: pd.DataFrame) -> Any:
    """Standardise tabular features (legacy CSV path)."""
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def split_data(data: Any, train_size: float = 0.8) -> Tuple[Any, Any]:
    """Split tabular rows (legacy CSV path)."""
    return train_test_split(data, train_size=train_size)


def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """Placeholder feature extraction for tabular data."""
    return data.drop(columns=["target"])


def load_and_preprocess(file_path: str) -> Any:
    """Load + standardise tabular data (legacy CSV path)."""
    return preprocess_data(load_data(file_path))


def validate_graph(data: Data, require_masks: bool = False) -> None:
    """Fail loudly if a graph is internally inconsistent.

    This catches the failure mode that produced
    ``IndexError: index N is out of bounds ...`` on CPU and
    ``CUDA error: device-side assert triggered`` on GPU: an ``edge_index`` whose
    node ids reach past the number of rows in ``x``. Doing the check here means
    it runs once, on the host, with a message that names the actual problem.
    """
    if not isinstance(data, Data):
        raise TypeError(f"Expected a torch_geometric Data object, got {type(data)!r}.")
    if data.x is None:
        raise ValueError("Graph has no node features (data.x).")
    if data.edge_index is None:
        raise ValueError("Graph has no edges (data.edge_index).")
    if data.y is None:
        raise ValueError("Graph has no targets (data.y).")
    if data.edge_index.dtype != torch.long:
        raise ValueError(f"edge_index must be int64, got {data.edge_index.dtype}.")

    num_nodes = int(data.num_nodes)
    if data.edge_index.numel():
        low = int(data.edge_index.min())
        high = int(data.edge_index.max())
        if low < 0 or high >= num_nodes:
            raise ValueError(
                f"edge_index references node ids outside [0, {num_nodes}) "
                f"(min={low}, max={high}). The graph was most likely subsetted "
                "without reindexing its edges; use masks instead of removing nodes."
            )
    if data.y.size(0) != num_nodes:
        raise ValueError(
            f"y has {data.y.size(0)} rows but the graph has {num_nodes} nodes."
        )
    if require_masks:
        for split in SPLITS:
            mask = getattr(data, f"{split}_mask", None)
            if mask is None:
                raise ValueError(f"Graph has no {split}_mask; run test/planetoid_split_save.py.")
            if mask.dtype != torch.bool:
                raise ValueError(f"{split}_mask must be boolean, got {mask.dtype}.")
            if mask.size(0) != num_nodes:
                raise ValueError(
                    f"{split}_mask has {mask.size(0)} entries, graph has {num_nodes} nodes."
                )


def load_pyg_data(file_path: str, weights_only: bool = False, require_masks: bool = False) -> Data:
    """Load a ``Data`` object from ``.pt`` and validate it.

    ``weights_only=False`` is required for PyG ``Data`` objects (torch >= 2.6
    defaults to ``True`` and refuses to unpickle them).
    """
    data = torch.load(file_path, weights_only=weights_only)
    validate_graph(data, require_masks=require_masks)
    return data


def normalize_node_features(data: Data) -> Data:
    """Standardise node features in place (fit on all rows; not for eval-only use)."""
    if data.x is None:
        raise ValueError("Graph has no node features (data.x) to normalise.")
    scaler = StandardScaler()
    # float64 fit/transform so the standardiser works on binary or integer
    # feature blocks too; the graph stays float32.
    x_norm = scaler.fit_transform(data.x.cpu().numpy().astype("float64"))
    data.x = torch.as_tensor(x_norm, dtype=torch.float32)
    return data


def graph_summary(data: Data) -> dict:
    """Small dict of shapes/counts, printed at start-up and in the data builder."""
    summary = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)) if data.edge_index is not None else 0,
        "num_node_features": int(data.num_node_features),
        "edge_index_max": int(data.edge_index.max()) if data.edge_index is not None and data.edge_index.numel() else -1,
    }
    if data.y is not None:
        summary["num_targets"] = int(data.y.size(0))
        summary["target_dim"] = int(data.y.size(-1)) if data.y.dim() > 1 else 1
        if data.y.dtype != torch.float32:
            summary["num_classes"] = int(data.y.max().item()) + 1
    for split in SPLITS:
        mask = getattr(data, f"{split}_mask", None)
        if mask is not None:
            summary[f"{split}_nodes"] = int(mask.sum().item())
    summary["is_undirected"] = bool(data.is_undirected())
    return summary


# ---------------------------------------------------------------------------
# Design notes for the CLM5 emulator data pipeline
# ---------------------------------------------------------------------------
# We want to read CLM5 surface data, forcing data, parameter data as predictors
# and the restart file as target.
#
# Open questions:
# - PFT level parameters
#   - they are graph-level, but graph-level features are implemented by copying
#     them onto node level
#   - are they allowed to be sparse or repeated along the node feature dims?
#   - it does not make sense to take the most important parameters from the
#     one-at-a-time experiment study, because it only represents the most
#     important parameters in terms of when they are changed
#   - rather, the parameters that make the most difference in the PFT definition
