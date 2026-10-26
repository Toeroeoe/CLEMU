import torch
import yaml
import os
from torch_geometric.loader import DataLoader 
from models.gnn import GNN
from data.preprocess import load_pyg_data
from utils.helpers import get_optimizer, get_loss_function

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int
) -> None:
    
    """
    Train the GNN model on the provided data loader.
    Args:
        model (torch.nn.Module): The GNN model to train.
        train_loader (DataLoader): DataLoader for training data.
        loss_function (torch.nn.Module): Loss function to use.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the training on (CPU or GPU).
        num_epochs (int): Number of epochs to train for.
    Returns:
        None
    It is assumed that the model, data, and loss function are compatible.
    The model is trained in a distributed setting using PyTorch's DistributedDataParallel.
    It is set up to run on multiple GPUs if available.
    """

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.x, batch.edge_index)
            # Node-level targets
            targets = batch.y if hasattr(batch, 'y') else None
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def main() -> None:
    torch.distributed.init_process_group(backend="nccl")  # Use "nccl" for GPUs
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or not config:
        raise RuntimeError("Config file 'config.yaml' is empty or invalid. Please provide a valid YAML config.")

    print("Loaded config:", config)
    train_cfg = config["train"]
    model_cfg = config["model"]

    # Load PyG graph data from .pt file
    train_data = load_pyg_data(train_cfg["train_data_path"],
                               weights_only=False)

    train_loader = DataLoader(
        [train_data],
        batch_size=train_cfg["batch_size"],
        shuffle=True
    )

    # Model parameters
    input_dim = model_cfg["input_dim"]
    output_dim = model_cfg["output_dim"]
    hidden_dim = model_cfg["hidden_dim"]
    num_layers = model_cfg["num_layers"]
    aggregation = model_cfg["aggregation"]
    dropout = model_cfg["dropout"]
    layer_type = model_cfg["layer_type"]
    activation = model_cfg["activation"]

    # Initialize model, loss function, and optimizer
    model = GNN(input_dim, 
                output_dim,
                hidden_dim, 
                num_layers, 
                aggregation=aggregation, 
                dropout=dropout,
                layer_type=layer_type,
                activation=activation).to(device)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    loss_function = get_loss_function(train_cfg["loss_function"])
    optimizer_cfg = train_cfg["optimizer"]
    optimizer = get_optimizer(
        optimizer_cfg["type"],
        model.parameters(),
        lr=optimizer_cfg["learning_rate"],
        **{k: v for k, v in optimizer_cfg.items() if k not in ["type", "learning_rate"]}
    )

    # Train the model
    train_model(model, train_loader, loss_function, optimizer, device, train_cfg["num_epochs"])

    # Save the trained model (only on rank 0)
    if torch.distributed.get_rank() == 0:
        torch.save(model.module.state_dict(), train_cfg["model_save_path"])
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()