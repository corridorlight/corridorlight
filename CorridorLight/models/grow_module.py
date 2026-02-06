"""
Grow Module: sequential rollout policy.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class GrowModule(nn.Module):
    """
    Sequential rollout policy that grows a corridor from a seed.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        max_neighbors: int = 4,
        max_length: int = 10
    ):
        """Create a Grow module."""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_neighbors = max_neighbors
        self.max_length = max_length
        self.action_dim = max_neighbors + 1  # neighbors + STOP
        
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),  # state + candidate node embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # score for one candidate action
        )
        
        self.q_value_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),  # state + candidate node embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q(s, a)
        )
        
        self.preference_head = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # preference in [0, 1]
        )
        nn.init.xavier_uniform_(self.preference_head[2].weight, gain=0.5)
        nn.init.zeros_(self.preference_head[2].bias)
    
    def encode_state(self, corridor_prefix: List[int], node_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode a corridor prefix into a fixed-size state."""
        if len(corridor_prefix) == 0:
            return torch.zeros(self.state_encoder[0].in_features)
        
        prefix_embeddings = node_embeddings[corridor_prefix]  # [prefix_len, embedding_dim]
        
        avg_embedding = prefix_embeddings.mean(dim=0)  # [embedding_dim]
        
        state = self.state_encoder(avg_embedding)
        return state
    
    def forward(
        self,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        action_mask: torch.Tensor,
        return_q_values: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, ...]:
        """Compute action logits (and optional Q-values) over candidates + STOP."""
        num_candidates = candidate_embeddings.shape[0]
        dev = state.device
        state_exp = state.unsqueeze(0).expand(num_candidates, -1)  # [N, hidden]
        combined = torch.cat([state_exp, candidate_embeddings], dim=-1)  # [N, hidden+emb]
        action_logits = self.policy_head(combined).squeeze(-1)  # [N]
        preference_values = self.preference_head(combined).squeeze(-1)  # [N]
        if return_q_values:
            q_cand = self.q_value_head(combined).squeeze(-1)  # [N]
        stop_logit = torch.tensor(0.0, device=dev, dtype=action_logits.dtype)
        action_logits = torch.cat([action_logits, stop_logit.unsqueeze(0)], dim=0)  # [N+1]
        if return_q_values:
            stop_q = torch.tensor(0.0, device=dev, dtype=q_cand.dtype)
            q_values = torch.cat([q_cand, stop_q.unsqueeze(0)], dim=0)  # [N+1]
        
        if action_mask.device != action_logits.device:
            action_mask = action_mask.to(action_logits.device)
        
        if action_mask.shape[0] != action_logits.shape[0]:
            raise ValueError(
                f"action_mask length ({action_mask.shape[0]}) must equal action_logits length ({action_logits.shape[0]})"
            )
        
        action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))
        if return_q_values:
            q_values = q_values.masked_fill(action_mask == 0, float('-inf'))
        
        if return_q_values:
            return action_logits, preference_values, q_values
        return action_logits, preference_values

