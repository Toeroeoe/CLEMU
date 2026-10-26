from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, Any
import torch
from torch_geometric.data import Data

def load_data(file_path: str) -> pd.DataFrame:
    # Function to load data from a specified file path
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data: pd.DataFrame) -> Any:
    # Function to preprocess the data
    # This may include normalization, handling missing values, etc.
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def split_data(data: Any, train_size: float = 0.8) -> Tuple[Any, Any]:
    # Function to split the data into training and testing sets
    train_data, test_data = train_test_split(data, train_size=train_size)
    return train_data, test_data

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    # Function to extract relevant features from the data
    # This is a placeholder for feature extraction logic
    features = data.drop(columns=['target'])  # Assuming 'target' is the label column
    return features

def load_and_preprocess(file_path: str) -> Any:
    # Combined function to load and preprocess data
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    return processed_data

def load_pyg_data(file_path: str, weights_only: bool = True) -> Data:
    # Load a PyTorch Geometric Data object from a .pt file
    data = torch.load(file_path, weights_only=weights_only)
    return data


# We want to read CLM5 surface data, forcing data, parameter data as predictors
# And restart file as target
# Make a first design of the graph structure with all teh variables
# Open questions:
# - PFT level parameters
#   - they are graph-level, but graph level are implemented by copying on node-level
#   - are they allowed to be sparse or repeated along the nodel feature dimensions?
#   - It does not make sens to take the most impotatnt paramters from the One at a time experiment study
#   - Because it only represents the most imporatnt parameters in terms of when they are changed
#   - Rather, the parameters that make the most difference in the PFT definition