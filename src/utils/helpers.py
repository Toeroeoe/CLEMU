import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim

def prepare_dataset(data):
    if hasattr(data, "values"):
        data = data.values
    features = torch.tensor(data[:, :-1], dtype=torch.float32)
    targets = torch.tensor(data[:, -1], dtype=torch.float32).unsqueeze(1)
    return TensorDataset(features, targets)

def get_loss_function(name: str) -> nn.Module:
    if name == "MSELoss":
        return nn.MSELoss()
    elif name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    # Add more loss functions as needed
    else:
        raise ValueError(f"Unknown loss function: {name}")

def get_optimizer(name: str, parameters, lr: float, **kwargs):
    if name == "Adam":
        return optim.Adam(parameters, lr=lr, **kwargs)
    elif name == "SGD":
        return optim.SGD(parameters, lr=lr, **kwargs)
    # Add more optimizers as needed
    else:
        raise ValueError(f"Unknown optimizer: {name}")