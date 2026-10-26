import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import argparse
import os
from typing import cast

def normalize_graph_data(data: Data) -> Data:
    if not hasattr(data, 'x') or data.x is None:
        raise ValueError("Graph data has no node features (x) to normalize.")
    scaler = StandardScaler()
    x_np = data.x.cpu().numpy()
    x_norm = scaler.fit_transform(x_np)
    data.x = torch.tensor(x_norm, dtype=torch.float32)
    return data

def main():
    parser = argparse.ArgumentParser(description="Split and save Planetoid graph dataset for training and evaluation.")
    parser.add_argument('--dataset', 
                        type=str, 
                        default='Cora', 
                        choices=['Cora', 'CiteSeer', 'PubMed'], 
                        help='Planetoid dataset name')
    parser.add_argument('--split-ratio', 
                        type=float, 
                        default=0.8, 
                        help='Train split ratio (0-1); remainder is used for evaluation.')
    parser.add_argument('--normalize', 
                        action='store_true', 
                        default=False,
                        help='Normalize node features')
    parser.add_argument('--out-dir', 
                        type=str, 
                        default='.', 
                        help='Output directory')
    args = parser.parse_args()

    # Load Planetoid dataset
    dataset = Planetoid(root=os.path.join(args.out_dir, args.dataset), name=args.dataset)
    data = cast(Data, dataset[0])

    if args.normalize:
        data = normalize_graph_data(data)
    
    if not hasattr(data, 'x') or data.x is None:
        raise ValueError("Normalized graph data has no node features.")
    
    if not hasattr(data, 'y') or not isinstance(data.y, torch.Tensor):
        raise ValueError("Normalized graph data has no target features.")
    
    # Split nodes into train/eval
    num_nodes = cast(int, data.num_nodes)
    num_train = int(num_nodes * args.split_ratio)
    perm = torch.randperm(num_nodes)
    train_idx = perm[:num_train]
    eval_idx = perm[num_train:]

    # Create train and eval Data objects
    train_data = Data(
        x=data.x[train_idx],
        edge_index=data.edge_index,
        y=data.y[train_idx],
        edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None
    )
    eval_data = Data(
        x=data.x[eval_idx],
        edge_index=data.edge_index,
        y=data.y[eval_idx],
        edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None
    )

    # Save to .pt files
    train_path = os.path.join(args.out_dir, f'{args.dataset}_train.pt')
    eval_path = os.path.join(args.out_dir, f'{args.dataset}_eval.pt')
    torch.save(train_data, train_path)
    torch.save(eval_data, eval_path)
    print(f"Saved train data to {train_path}")
    print(f"Saved evaluation data to {eval_path}")

if __name__ == "__main__":
    main()
