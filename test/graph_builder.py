import torch
from torch_geometric.data import Data

def build_directed_example_graph():
    
    # Example: 4 nodes, each with 3 features
    x = torch.tensor([
        [0.1, 0.2, 0.3],  # Node 0 features
        [0.4, 0.5, 0.6],  # Node 1 features
        [0.7, 0.8, 0.9],  # Node 2 features
        [1.0, 1.1, 1.2],  # Node 3 features
    ], dtype=torch.float)

    # Directed edges: only one direction for each pair
    edge_index = torch.tensor([
        [0, 2, 0],  # source nodes
        [1, 3, 2],  # target nodes
    ], dtype=torch.long)

    # Optional: edge features
    edge_attr = torch.tensor([
        [1.0],  # Edge 0->1
        [0.5],  # Edge 2->3
        [2.0],  # Edge 0->2
    ], dtype=torch.float)

    # Optional: node labels or targets
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    # Create the graph data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

# Example usage:
if __name__ == "__main__":
    graph = build_directed_example_graph()

    print(graph.is_directed())  # Should print False
    print(graph)