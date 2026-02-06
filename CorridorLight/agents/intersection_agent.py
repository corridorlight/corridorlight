"""
Intersection Agent (lower-level intersection signal agent).
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


class SharedBackbone(nn.Module):
    """Shared backbone network."""
    
    def __init__(self, observation_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.backbone(observation)


class LocalHead(nn.Module):
    """Local policy head."""
    
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class GlobalHead(nn.Module):
    """Global policy head (receives corridor-task-related observations)."""
    
    def __init__(self, hidden_dim: int, action_dim: int, corridor_task_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + corridor_task_dim, hidden_dim),  # backbone features + task vector
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        corridor_task_vec: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([features, corridor_task_vec], dim=-1)
        return self.head(combined)


class ValueHead(nn.Module):
    """Value function head."""
    
    def __init__(self, hidden_dim: int, global_input_dim: int = 0):
        super().__init__()
        input_dim = hidden_dim + global_input_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features: torch.Tensor, global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if global_features is not None:
            features = torch.cat([features, global_features], dim=-1)
        return self.head(features).squeeze(-1)


class DeltaHead(nn.Module):
    """Predict standardized advantage gap Delta for gating.

    Paper-aligned gate:
      Delta = \\hat A^{coop} - \\hat A^{loc},
      lambda* = Pi_{[0,1]}(Delta / rho).

    In implementation, Delta is predicted from (o_t, G_t) and trained with supervised targets
    computed from GAE advantages during PPO updates.
    """

    def __init__(self, hidden_dim: int, corridor_task_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + corridor_task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, corridor_task_vec: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([features, corridor_task_vec], dim=-1)
        return self.head(combined).squeeze(-1)


class IntersectionAgent:
    """
    Lower-level intersection signal agent (local vs corridor-aware policy mixing).
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        lane_feature_dim: int,
        config: Dict[str, Any],
        lane_ids: Optional[list[str]] = None,
        corridor_task_dim: Optional[int] = None,
        shared_modules: Optional[Dict[str, nn.Module]] = None
    ):
        """Create an intersection agent."""
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.lane_feature_dim = lane_feature_dim
        self.config = config
        self.lane_ids: list[str] = list(lane_ids) if lane_ids else []
        if corridor_task_dim is not None:
            target_dim = int(corridor_task_dim)
            if target_dim < 0:
                target_dim = 0
            target_lanes = int(target_dim // 2)
            if target_lanes > 0:
                if len(self.lane_ids) < target_lanes:
                    pad_n = target_lanes - len(self.lane_ids)
                    self.lane_ids = self.lane_ids + [f"__pad_lane__{i}" for i in range(pad_n)]
                elif len(self.lane_ids) > target_lanes:
                    self.lane_ids = self.lane_ids[:target_lanes]
            self.corridor_task_dim = int(2 * len(self.lane_ids))
        else:
            self.corridor_task_dim = int(2 * len(self.lane_ids))
        
        device_str = config.get("device", "cpu")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        
        hidden_dim = config.get("hidden_dim", 128)
        self.lambda_friction_coeff = config.get("lambda_friction_coeff", 1.0)
        self.lambda_mode = str(config.get("lambda_mode", "advantage_gap")).lower()
        
        self.include_grow_q_in_global_critic = bool(config.get("include_grow_q_in_global_critic", False))
        critic_extra_dim = 1 if self.include_grow_q_in_global_critic else 0

        if shared_modules is not None:
            self.backbone = shared_modules["backbone"]
            self.local_head = shared_modules["local_head"]
            self.global_head = shared_modules["global_head"]
            self.value_loc_head = shared_modules["value_loc_head"]
            self.value_glob_head = shared_modules["value_glob_head"]
            # Optional module (may be None for older checkpoints/configs)
            self.delta_head = shared_modules.get("delta_head", None) if isinstance(shared_modules, dict) else None
            try:
                self.device = next(self.backbone.parameters()).device
            except Exception:
                pass
        else:
            self.backbone = SharedBackbone(observation_dim, hidden_dim).to(self.device)
            self.local_head = LocalHead(hidden_dim, action_dim).to(self.device)
            self.global_head = GlobalHead(hidden_dim, action_dim, self.corridor_task_dim).to(self.device)
            self.value_loc_head = ValueHead(hidden_dim).to(self.device)
            self.value_glob_head = ValueHead(hidden_dim, global_input_dim=self.corridor_task_dim + critic_extra_dim).to(self.device)
            self.delta_head = DeltaHead(hidden_dim, self.corridor_task_dim).to(self.device)
        
        self.preference_values = {}  # lane_id -> preference value
        self.lane_mask = {}  # lane_id -> bool (whether it belongs to the corridor)
        self.corridor_q_value = 0.0
        
        self.disable_corridor_agent = config.get("disable_corridor_agent", False)
    
    def set_corridor_task(
        self,
        lane_mask: Dict[str, bool],
        preference_values: Dict[str, float],
        corridor_q_value: float = 0.0
    ):
        """Set lane_mask/preferences provided by the corridor agent."""
        self.lane_mask = lane_mask
        self.preference_values = preference_values
        try:
            self.corridor_q_value = float(corridor_q_value)
        except Exception:
            self.corridor_q_value = 0.0

    def _build_global_critic_features(self, lane_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build global features for the cooperation-task critic (task_dim (+1))."""
        task_vec = self._build_corridor_task_vec(lane_features)
        if not getattr(self, "include_grow_q_in_global_critic", False):
            return task_vec
        v = float(getattr(self, "corridor_q_value", 0.0))
        if not np.isfinite(v):
            v = 0.0
        v_t = torch.tensor([v], dtype=task_vec.dtype, device=task_vec.device)
        return torch.cat([task_vec, v_t], dim=-1)

    def _build_corridor_task_vec(self, lane_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten [mask..., pref...] aligned to lane_ids."""
        lane_ids = self.lane_ids
        if not lane_ids:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        mask_part = [1.0 if (self.lane_mask or {}).get(lid, False) else 0.0 for lid in lane_ids]
        pref_part = [float((self.preference_values or {}).get(lid, 0.0)) for lid in lane_ids]
        vec = mask_part + pref_part
        t = torch.tensor(vec, dtype=torch.float32, device=self.device)
        return t
    
    def compute_lambda(self, observation: torch.Tensor, lane_features: Dict[str, torch.Tensor]) -> float:
        """
        Compute the participation gate lambda* in [0, 1].
        """
        backbone_features = self.backbone(observation)

        rho = float(getattr(self, "lambda_friction_coeff", 1.0))
        if rho <= 0.0:
            rho = 1.0

        mode = str(getattr(self, "lambda_mode", "advantage_gap")).lower()
        if mode == "advantage_gap" and getattr(self, "delta_head", None) is not None:
            # Paper-aligned gate uses Delta = \hat A^{coop} - \hat A^{loc} (standardized advantages).
            # At runtime we estimate Delta from (o_t, G_t) using a learned predictor delta_head.
            task_vec = self._build_corridor_task_vec(lane_features)
            try:
                if backbone_features.dim() == 1 and task_vec.dim() == 2 and task_vec.shape[0] == 1:
                    task_vec = task_vec.squeeze(0)
                elif backbone_features.dim() == 2 and task_vec.dim() == 1:
                    task_vec = task_vec.unsqueeze(0)
            except Exception:
                pass

            try:
                delta_hat = self.delta_head(backbone_features, task_vec)
                if not torch.isfinite(delta_hat).all():
                    return 0.0
            except Exception:
                return 0.0

            lambda_star = torch.clamp(delta_hat / rho, 0.0, 1.0).item()
        else:
            # Fallback (legacy): value-based normalized gap.
            v_loc = self.value_loc_head(backbone_features)
            critic_features = self._build_global_critic_features(lane_features)
            v_glob = self.value_glob_head(backbone_features, critic_features)
            try:
                if not torch.isfinite(v_loc).all() or not torch.isfinite(v_glob).all():
                    return 0.0
            except Exception:
                return 0.0
            eps = float(self.config.get("lambda_delta_eps", 1e-6))
            denom = torch.abs(v_glob) + torch.abs(v_loc) + eps
            delta = (v_glob - v_loc) / denom
            lambda_star = torch.clamp(delta / rho, 0.0, 1.0).item()

        lambda_max_initial = self.config.get("lambda_max_initial", 1.0)  # Default: no cap.
        if lambda_max_initial < 1.0:
            lambda_star = min(lambda_star, lambda_max_initial)
        
        return lambda_star
    
    def _weight_lane_features(self, lane_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Weight lane features.
        
        f'_ℓ = (1 + p_ℓ) * f_ℓ
        """
        weighted_features = []
        for lane_id, features in lane_features.items():
            if self.lane_mask and not self.lane_mask.get(lane_id, False):
                continue
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            features = features.to(self.device)
            
            p_ell = self.preference_values.get(lane_id, 0.0)
            weighted = (1.0 + p_ell) * features
            weighted_features.append(weighted)
        
        if weighted_features:
            return torch.stack(weighted_features).mean(dim=0)
        else:
            return torch.zeros(self.lane_feature_dim, device=self.device)
    
    def select_action(
        self,
        observation: torch.Tensor,
        lane_features: Dict[str, torch.Tensor],
        training: bool = True,
        return_infos: bool = False,
        force_local_only: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action (mix-logits interpolation or single-objective mode).
        
        Args:
            observation: Observation.
            lane_features: lane id -> features
            training: Whether in training mode.
            return_infos: Whether to also return log_prob and value (avoid recomputation during training).
            
        Returns:
            action: Selected action.
            info: Extra info (lambda, logits, log_prob, value, etc.).
        """
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.to(self.device)
        
        backbone_features = self.backbone(observation)
        
        local_logits = self.local_head(backbone_features)
        
        if self.disable_corridor_agent or force_local_only:
            dist = torch.distributions.Categorical(logits=local_logits)
            
            if training:
                action = dist.sample().item()
            else:
                action = torch.argmax(local_logits).item()
            
            value_loc = self.value_loc_head(backbone_features)
            
            info = {
                "lambda": 0.0,
                "local_logits": local_logits.detach().cpu().numpy(),
                "global_logits": None,
                "mixed_logits": local_logits.detach().cpu().numpy(),
                "mode": "single_objective" if self.disable_corridor_agent else "forced_local_only"
            }
            
            if return_infos:
                action_tensor = torch.tensor(action, device=self.device)
                log_prob = dist.log_prob(action_tensor)
                info["log_prob"] = log_prob
                info["value_loc"] = value_loc
                info["value_glob"] = value_loc  # Same in single-objective mode.
            
            return action, info
        
        if (not self.lane_mask) or (not any(self.lane_mask.values())):
            dist = torch.distributions.Categorical(logits=local_logits)
            if training:
                action = dist.sample().item()
            else:
                action = torch.argmax(local_logits).item()
            value_loc = self.value_loc_head(backbone_features)
            info = {
                "lambda": 0.0,
                "local_logits": local_logits.detach().cpu().numpy(),
                "global_logits": None,
                "mixed_logits": local_logits.detach().cpu().numpy(),
                "mode": "forced_local_only_no_corridor"
            }
            if return_infos:
                action_tensor = torch.tensor(action, device=self.device)
                log_prob = dist.log_prob(action_tensor)
                info["log_prob"] = log_prob
                info["value_loc"] = value_loc
                info["value_glob"] = value_loc
            return action, info
        
        lambda_star = self.compute_lambda(observation, lane_features)
        try:
            if not np.isfinite(float(lambda_star)):
                lambda_star = 0.0
        except Exception:
            lambda_star = 0.0
        lambda_star = float(max(0.0, min(1.0, float(lambda_star))))
        
        task_vec = self._build_corridor_task_vec(lane_features)
        try:
            if backbone_features.dim() == 1 and task_vec.dim() == 2 and task_vec.shape[0] == 1:
                task_vec = task_vec.squeeze(0)
            elif backbone_features.dim() == 2 and task_vec.dim() == 1:
                task_vec = task_vec.unsqueeze(0)
        except Exception:
            pass
        global_logits = self.global_head(backbone_features, task_vec)
        
        # Sanitize logits before feeding into Categorical.
        local_logits = torch.nan_to_num(local_logits, nan=0.0, posinf=0.0, neginf=0.0)
        global_logits = torch.nan_to_num(global_logits, nan=0.0, posinf=0.0, neginf=0.0)
        mixed_logits = (1 - lambda_star) * local_logits + lambda_star * global_logits
        mixed_logits = torch.nan_to_num(mixed_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(mixed_logits).all():
            mixed_logits = torch.zeros_like(local_logits)
        
        dist = torch.distributions.Categorical(logits=mixed_logits)
        if training:
            action = dist.sample().item()
        else:
            action = torch.argmax(mixed_logits).item()
        
        info = {
            "lambda": lambda_star,
            "local_logits": local_logits.detach().cpu().numpy(),
            "global_logits": global_logits.detach().cpu().numpy(),
            "mixed_logits": mixed_logits.detach().cpu().numpy(),
            "mode": "multi_objective"
        }
        
        if return_infos:
            action_tensor = torch.tensor(action, device=self.device)
            log_prob = dist.log_prob(action_tensor)
            value_loc = self.value_loc_head(backbone_features)
            value_glob = self.value_glob_head(backbone_features, self._build_global_critic_features(lane_features))
            info["log_prob"] = log_prob
            info["value_loc"] = value_loc
            info["value_glob"] = value_glob
        
        return action, info
    
    def evaluate_action(
        self,
        observation: torch.Tensor,
        action: int,
        lane_features: Dict[str, torch.Tensor],
        force_local_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate an action (get log_prob and value).
        
        Args:
            observation: Observation.
            action: Action.
            lane_features: lane id -> features
            
        Returns:
            log_prob: Log probability of the action.
            value_loc: Local value estimate.
            value_glob: Global value estimate (equals value_loc if corridor agent is disabled).
        """
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        observation = observation.to(self.device)
        
        backbone_features = self.backbone(observation)
        
        local_logits = self.local_head(backbone_features)
        local_dist = torch.distributions.Categorical(logits=local_logits)
        action_tensor = torch.tensor(action, device=self.device)
        log_prob_loc = local_dist.log_prob(action_tensor)
        
        value_loc = self.value_loc_head(backbone_features)
        
        if self.disable_corridor_agent or force_local_only:
            return log_prob_loc, value_loc, value_loc
        
        task_vec = self._build_corridor_task_vec(lane_features)
        try:
            if backbone_features.dim() == 1 and task_vec.dim() == 2 and task_vec.shape[0] == 1:
                task_vec = task_vec.squeeze(0)
            elif backbone_features.dim() == 2 and task_vec.dim() == 1:
                task_vec = task_vec.unsqueeze(0)
        except Exception:
            pass
        global_logits = self.global_head(backbone_features, task_vec)
        global_dist = torch.distributions.Categorical(logits=global_logits)
        log_prob_glob = global_dist.log_prob(action_tensor)
        
        value_glob = self.value_glob_head(backbone_features, self._build_global_critic_features(lane_features))
        
        lambda_star = self.compute_lambda(observation, lane_features)
        mixed_logits = (1 - lambda_star) * local_logits + lambda_star * global_logits
        mixed_dist = torch.distributions.Categorical(logits=mixed_logits)
        log_prob_mixed = mixed_dist.log_prob(action_tensor)
        
        return log_prob_mixed, value_loc, value_glob
    
    def save(self, filepath: str) -> None:
        """
        Save agent model parameters.
        
        Args:
            filepath: Output path.
        """
        import torch
        checkpoint = {
            'backbone_state_dict': self.backbone.state_dict(),
            'local_head_state_dict': self.local_head.state_dict(),
            'global_head_state_dict': self.global_head.state_dict(),
            'value_loc_head_state_dict': self.value_loc_head.state_dict(),
            'value_glob_head_state_dict': self.value_glob_head.state_dict(),
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'lane_feature_dim': self.lane_feature_dim,
            'config': self.config,
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load agent model parameters.
        
        Args:
            filepath: Checkpoint path.
        """
        import torch
        safe_np_globals = [np.core.multiarray.scalar, np.dtype, np.dtypes.Int64DType]
        with torch.serialization.safe_globals(safe_np_globals):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.local_head.load_state_dict(checkpoint['local_head_state_dict'])
        self.global_head.load_state_dict(checkpoint['global_head_state_dict'])
        self.value_loc_head.load_state_dict(checkpoint['value_loc_head_state_dict'])
        self.value_glob_head.load_state_dict(checkpoint['value_glob_head_state_dict'])

