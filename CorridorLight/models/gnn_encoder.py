"""
GNN encoder module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GNNEncoder(nn.Module):
    """
    GNN encoder module.

    Performs global message passing on the direction graph to produce topology-aware embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        architecture: str = "GCN",
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension (e.g., pressure).
            hidden_dim: Hidden dimension.
            output_dim: Output embedding dimension.
            num_layers: Number of GNN layers.
            architecture: GNN architecture type (GCN, GAT, GraphSAGE).
            dropout: Dropout rate.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.architecture = architecture
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            if architecture == "GCN":
                conv = GCNConv(in_dim, out_dim)
            elif architecture == "GAT":
                conv = GATConv(in_dim, out_dim, heads=1, concat=False)
            elif architecture == "GraphSAGE":
                conv = SAGEConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unknown GNN architecture: {architecture}")
            
            self.convs.append(conv)
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data object (node features, edge_index).
            
        Returns:
            node embeddings: [num_nodes, output_dim]
        """
        x, edge_index = data.x, data.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.relu(x)
                x = self.dropout(x)
        
        return x

