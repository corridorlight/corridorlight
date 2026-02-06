"""
Seed Module: global retriever (Top-k retrieval).
"""

import torch
import torch.nn as nn


class SeedModule(nn.Module):
    """
    Seed Module: state-value function V(s).

    Estimates the expected return starting from a node (state), i.e. V(seed).
    This is a state-value function and does not depend on a specific action.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        """
        Args:
            embedding_dim: GNN embedding dimension (state representation).
            hidden_dim: Hidden dimension of the value head.
        """
        super().__init__()
        # State-value function V(s): input is state (node embedding), output is value.
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output V(s)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the state value V(s).
        
        Args:
            embeddings: [num_nodes, embedding_dim] Node state representations.
            
        Returns:
            seed_values: [num_nodes] State values V(s) for each node.
        """
        return self.value_head(embeddings).squeeze(-1)
    
    def select_top_k(self, embeddings: torch.Tensor, k: int) -> list:
        """
        Select Top-k candidate seeds.
        
        Args:
            embeddings: [num_nodes, embedding_dim]
            k: Number of Top-k seeds.
            
        Returns:
            top_k_indices: List of Top-k node indices.
        """
        seed_values = self.forward(embeddings)
        _, top_k_indices = torch.topk(seed_values, k)
        return top_k_indices.tolist()

