import torch
import numpy as np
from torch_geometric.data import Data

def random_directed_graph(num_nodes=1000, num_features=10, num_edges=3000, seed=42, undirected_ratio=0.2):
    np.random.seed(seed)
    # Node features
    x = torch.tensor(np.random.randn(num_nodes, num_features), dtype=torch.float32)
    # Node targets (regression)
    y = torch.tensor(np.random.randn(num_nodes, 1), dtype=torch.float32)

    # Directed and undirected edge index generation
    edge_list = []
    for _ in range(num_edges):
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        edge_list.append([src, dst])
        # Add undirected edge with some probability
        if np.random.rand() < undirected_ratio and src != dst:
            edge_list.append([dst, src])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # shape [2, num_edges]

    # Optional: edge features
    edge_attr = torch.tensor(np.random.rand(edge_index.shape[1], 1), dtype=torch.float32)

    # Build PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

def save_graph(graph, path):
    torch.save(graph, path)
    print(f"Directed graph saved to {path}")

def main():
    graph = random_directed_graph(
        num_nodes=1000,
        num_features=10,
        num_edges=3000,
        seed=42,
        undirected_ratio=0.2  # 20% of edges are undirected (added both ways)
    )
    save_graph(graph, "directed_synthetic_graph.pt")
    print(graph)
    print("Edge index shape:", graph.edge_index.shape)
    print("Node feature shape:", graph.x.shape)
    print("Target shape:", graph.y.shape)
    print("Edge feature shape:", graph.edge_attr.shape)

if __name__ == "__main__":
    main()