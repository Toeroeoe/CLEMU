"""Shared helpers: device selection, dimension inference, loss/optimizer factories.

Two rules here exist because both were silent or confusing failure modes in the
first implementation:

* ``input_dim`` / ``output_dim`` may be omitted from the config. When they are,
  they are inferred from the loaded graph instead of trusted blindly -- a config
  that says ``input_dim: 10`` while ``x`` has 1433 columns fails deep inside the
  first ``GCNConv`` with a matmul shape error that says nothing about the config.
* Targets are reshaped to the model output before the loss is computed.
  ``MSELoss(outputs[N, 1], y[N])`` broadcasts to ``[N, N]`` without raising, so
  a wrong target shape trains happily on nonsense.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Pick a device without failing on hosts that have no GPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return "cpu"


def infer_output_dim(data, task: str) -> int:
    """Infer the model output width from the targets in ``data``."""
    y = data.y
    if y is None:
        raise ValueError("Graph has no targets (data.y); cannot infer output_dim.")
    if task == "classification":
        if y.dim() > 1 and y.size(-1) > 1:
            return int(y.size(-1))
        return int(y.max().item()) + 1
    if task == "regression":
        return int(y.size(-1)) if y.dim() > 1 else 1
    raise ValueError(f"Unknown task: {task!r} (expected 'classification' or 'regression').")


def resolve_dimensions(data, model_cfg: dict, task: str) -> tuple:
    """Resolve ``(input_dim, output_dim)`` from the config, falling back to the data."""
    input_dim = model_cfg.get("input_dim")
    if not input_dim:
        if data.x is None:
            raise ValueError("Graph has no node features (data.x); cannot infer input_dim.")
        input_dim = int(data.x.size(-1))
    output_dim = model_cfg.get("output_dim")
    if not output_dim:
        output_dim = infer_output_dim(data, task)
    return int(input_dim), int(output_dim)


def prepare_targets(y: torch.Tensor, outputs: torch.Tensor, task: str) -> torch.Tensor:
    """Shape and cast targets so they match ``outputs``."""
    if task == "classification":
        # CrossEntropyLoss wants [N] of dtype long against logits [N, C].
        return y.reshape(-1).to(torch.long)
    if task == "regression":
        # MSELoss/L1Loss want the same trailing shape as the prediction.
        return y.reshape(-1, outputs.size(-1)).to(outputs.dtype)
    raise ValueError(f"Unknown task: {task!r} (expected 'classification' or 'regression').")


def get_loss_function(name: str) -> nn.Module:
    if name == "MSELoss":
        return nn.MSELoss()
    if name == "L1Loss":
        return nn.L1Loss()
    if name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss function: {name}")


def get_optimizer(name: str, parameters, lr: float, **kwargs) -> optim.Optimizer:
    if name == "Adam":
        return optim.Adam(parameters, lr=lr, **kwargs)
    if name == "AdamW":
        return optim.AdamW(parameters, lr=lr, **kwargs)
    if name == "SGD":
        return optim.SGD(parameters, lr=lr, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def ensure_parent_dir(path: str) -> str:
    """Create the parent directory of ``path`` so ``torch.save`` cannot fail late."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prepare_dataset(data) -> TensorDataset:
    """Legacy tabular helper (features + last column as target)."""
    if hasattr(data, "values"):
        data = data.values
    features = torch.tensor(data[:, :-1], dtype=torch.float32)
    targets = torch.tensor(data[:, -1], dtype=torch.float32).unsqueeze(1)
    return TensorDataset(features, targets)
