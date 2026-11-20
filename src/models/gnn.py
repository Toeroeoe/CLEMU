import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch import Tensor

class GNN(nn.Module):

    """
    Graph Neural Network model with configurable layers and activation functions.
    Supports GCN, SAGE, and GAT layers.
    For now, designed for node-level regression and classification tasks.
    """

    # Layer classes that take an explicit `aggr`: GATConv aggregates through its
    # attention mechanism and ignores `aggr`, but would accept it via **kwargs
    # and silently do nothing.
    AGGREGATING_LAYERS = ('GCNConv', 'SAGEConv')

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        aggregation: str = 'mean',
        dropout: float = 0.0,
        layer_type: str = 'GCNConv',
        activation: str = 'relu'
    ) -> None:
        super(GNN, self).__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2 (input and output layer required).")
        self.aggregation = aggregation
        self.dropout = dropout
        self.layer_type = layer_type
        self.activation = activation
        self.convs = nn.ModuleList()
        # Select layer class
        layer_cls = self._get_layer_class(self.layer_type)
        # Input layer
        conv_kwargs = {"aggr": aggregation} if self.layer_type in self.AGGREGATING_LAYERS else {}
        self.convs.append(layer_cls(input_dim, hidden_dim, **conv_kwargs))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(layer_cls(hidden_dim, hidden_dim, **conv_kwargs))
        # Output layer
        self.convs.append(layer_cls(hidden_dim, output_dim, **conv_kwargs))

    def _get_layer_class(self, layer_type: str):
        if layer_type == 'GCNConv':
            return GCNConv
        elif layer_type == 'SAGEConv':
            return SAGEConv
        elif layer_type == 'GATConv':
            return GATConv
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")

    def _get_activation(self, activation: str):
        if activation == 'relu':
            return F.relu
        elif activation == 'leaky_relu':
            return F.leaky_relu
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 'none' or activation is None:
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, 
                x: Tensor, 
                edge_index: Tensor) -> Tensor:
        """
        Forward pass for the GNN.
        If graph_features is provided, concatenate it to each node's features.
        Args:
            x (Tensor): Node features [num_nodes, input_dim]
            edge_index (Tensor): Edge indices
        Returns:
            Tensor: Node-level outputs [num_nodes, output_dim]
        """

        act_fn = self._get_activation(self.activation)

        for conv in self.convs[:-1]:
            x = act_fn(conv(x, edge_index))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x