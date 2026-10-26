import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import yaml
from typing import Callable, Tuple, Any, Dict
from models.gnn import GNN
from data.preprocess import load_data, preprocess_data
from utils.helpers import get_loss_function, prepare_dataset

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples

    return average_loss, accuracy

def visualize_results(results: Dict[str, Any]) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(results['epochs'], results['loss'], label='Loss')
    plt.plot(results['epochs'], results['accuracy'], label='Accuracy')
    plt.title('Model Evaluation Results')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()

def main() -> None:
    # Load config
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or not config:
        raise RuntimeError("Config file '../config.yaml' is empty or invalid. Please provide a valid YAML config.")
    
    eval_cfg = config["evaluate"]
    model_cfg = config["model"]

    # Load and preprocess eval data
    eval_data = load_data(eval_cfg["eval_data_path"])
    eval_data = preprocess_data(eval_data)
    dataset = prepare_dataset(eval_data)
    eval_loader = DataLoader(dataset, batch_size=eval_cfg["batch_size"], shuffle=False)

    # Model parameters
    input_dim = model_cfg["input_dim"]
    output_dim = model_cfg["output_dim"]
    hidden_dim = model_cfg["hidden_dim"]
    num_layers = model_cfg["num_layers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(input_dim, output_dim, hidden_dim, num_layers).to(device)
    model.load_state_dict(torch.load(eval_cfg["model_load_path"], map_location=device))

    loss_function = get_loss_function(eval_cfg["loss_function"])

    avg_loss, accuracy = evaluate_model(model, eval_loader, device, loss_function)
    print(f"Evaluation Loss: {avg_loss:.4f}, Evaluation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()