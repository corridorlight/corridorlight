"""
Corridor Agent (upper-level corridor agent).
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
import torch
from torch_geometric.data import Data

from ..models.gnn_encoder import GNNEncoder
from ..models.seed_module import SeedModule
from ..models.grow_module import GrowModule
from ..models.pressure_graph import PressureGraphBuilder


@dataclass
class Corridor:
    """Corridor data structure."""
    direction_nodes: List[int]
    seed_value: float
    preference_values: Dict[str, float]
    junctions: Set[str]
    stopped_immediately: bool = False


class CorridorAgent:
    """
    Upper-level corridor planner.
    """
    
    def __init__(
        self,
        gnn_encoder: GNNEncoder,
        seed_module: SeedModule,
        grow_module: GrowModule,
        config: Dict[str, Any]
    ):
        """Create a corridor agent."""
        self.gnn_encoder = gnn_encoder
        self.seed_module = seed_module
        self.grow_module = grow_module
        self.config = config
        
        self.max_corridors = config.get("max_corridors", 3)
        self.max_corridor_length = config.get("max_corridor_length", 10)
        self.seed_top_k = config.get("seed_top_k", 5)
        
        self.direction_graph = None
        self.node_to_junction = {}  # direction node id -> junction id
        self.direction_to_lanes = {}  # direction node id -> list of lane ids
    
    def set_direction_graph(self, pressure_graph: Data, graph_builder: PressureGraphBuilder):
        """Set the direction graph and related mappings."""
        self.direction_graph = pressure_graph
        self.node_to_junction = graph_builder.direction_to_junction.copy()
        self.direction_to_lanes = graph_builder.direction_to_lanes.copy()
    
    def generate_corridors(
        self,
        pressure_graph: Data,
        node_embeddings: torch.Tensor,
        return_log_probs: bool = False,
        return_q_values: bool = False,
        return_v_values: bool = False,
        return_pref_tensors: bool = False
    ):
        """
        Generate corridors from node embeddings.
        """
        self.direction_graph = pressure_graph
        
        corridors = []
        used_junctions = set()
        log_probs_list = []
        q_values_list = []
        v_values_list = []
        pref_tensors_list = []
        
        if return_v_values:
            seed_v_values = self.seed_module.forward(node_embeddings)  # [num_nodes]
            seed_scores = seed_v_values
        else:
            seed_scores = self.seed_module.forward(node_embeddings)
            seed_v_values = None
        ranked_seed_indices = torch.argsort(seed_scores, descending=True).tolist()
        
        num_grown_corridors = 0
        used_direction_nodes: Set[int] = set()
        dropped_seeds: Set[int] = set()
        considered_seeds = 0
        for seed_idx in ranked_seed_indices:
            if num_grown_corridors >= self.max_corridors:
                break
            if seed_idx in used_direction_nodes:
                continue
            if seed_idx in dropped_seeds:
                continue
            considered_seeds += 1
            if considered_seeds > self.seed_top_k:
                break
            
            result = self._grow_corridor(
                seed_idx,
                node_embeddings,
                used_junctions,
                return_log_probs=return_log_probs,
                return_q_values=return_q_values,
                return_pref_tensors=return_pref_tensors
            )
            
            if result is None:
                continue
            
            corridor = None
            log_probs = []
            q_vals = []
            pref_tensors = []

            if result is None:
                corridor = None
            elif (return_log_probs and return_q_values and return_pref_tensors):
                corridor, log_probs, q_vals, pref_tensors = result
            elif (return_log_probs and return_q_values):
                corridor, log_probs, q_vals = result
            elif (return_log_probs and return_pref_tensors):
                corridor, log_probs, pref_tensors = result
            elif (return_q_values and return_pref_tensors):
                corridor, q_vals, pref_tensors = result
            elif return_log_probs:
                corridor, log_probs = result
            elif return_q_values:
                corridor, q_vals = result
            elif return_pref_tensors:
                corridor, pref_tensors = result
            else:
                corridor = result

            if corridor is None:
                continue

            try:
                direction_nodes = list(getattr(corridor, "direction_nodes", []) or [])
            except Exception:
                direction_nodes = []
            if len(direction_nodes) < 2:
                dropped_seeds.add(int(seed_idx))
                continue

            num_grown_corridors += 1
            
            if getattr(corridor, "stopped_immediately", False):
                try:
                    direction_nodes = list(getattr(corridor, "direction_nodes", []))
                except Exception:
                    direction_nodes = []
                if len(direction_nodes) == 1:
                    lane_ids = set(self.direction_to_lanes.get(direction_nodes[0], []) or [])
                    if len(lane_ids) == 1:
                        break

            if isinstance(corridor, Corridor):
                try:
                    used_direction_nodes.update(list(corridor.direction_nodes))
                except Exception:
                    pass

            if isinstance(corridor, Corridor) and len(corridor.direction_nodes) > 1:
                corridors.append(corridor)
                if return_log_probs:
                    log_probs_list.append(log_probs)
                if return_q_values:
                    q_values_list.append(q_vals)
                if return_v_values:
                    v_values_list.append(seed_v_values[seed_idx])
                if return_pref_tensors:
                    pref_tensors_list.append(pref_tensors)
                used_junctions.update(corridor.junctions)
        
        result_list = [corridors]
        if return_log_probs:
            result_list.append(log_probs_list)
        if return_q_values:
            result_list.append(q_values_list)
        if return_v_values:
            result_list.append(v_values_list)
        if return_pref_tensors:
            result_list.append(pref_tensors_list)
        
        if len(result_list) > 1:
            return tuple(result_list)
        return corridors
    
    def _grow_corridor(
        self,
        seed_idx: int,
        node_embeddings: torch.Tensor,
        used_junctions: Set[str],
        return_log_probs: bool = False,
        return_q_values: bool = False,
        return_pref_tensors: bool = False
    ):
        """Grow a corridor starting from a seed."""
        corridor_prefix = [seed_idx]
        preference_values = {}
        log_probs = []  # Training: log_prob per action
        q_values = []  # Training: Q-value per action
        selected_pref_tensors = []  # Training: selected-action preference tensors (keeps grads)
        
        seed_junction = self.node_to_junction.get(seed_idx)
        if seed_junction and seed_junction in used_junctions:
            return None
        
        used_junctions_in_corridor = set()
        if seed_junction:
            used_junctions_in_corridor.add(seed_junction)
        
        stopped_immediately = False
        for step in range(self.max_corridor_length):
            state = self.grow_module.encode_state(corridor_prefix, node_embeddings)
            
            tail_node = corridor_prefix[-1]
            neighbors = self._get_neighbors(tail_node)
            
            if len(neighbors) == 0:
                break  # No neighbors; stop.
            
            candidate_indices = []
            candidate_embeddings = []
            action_mask = []
            
            for neighbor_idx in neighbors:
                if self._is_valid_action(corridor_prefix, neighbor_idx, used_junctions):
                    candidate_indices.append(neighbor_idx)
                    candidate_embeddings.append(node_embeddings[neighbor_idx])
                    action_mask.append(1)  # Mask only valid candidates.
            
            action_mask.append(1)  # STOP is always feasible.
            
            if len(candidate_indices) == 0:
                break  # No feasible actions.
            
            candidate_embeddings_tensor = torch.stack(candidate_embeddings)
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=node_embeddings.device)
            
            if return_q_values:
                action_logits, pref_values, q_vals = self.grow_module(
                    state,
                    candidate_embeddings_tensor,
                    action_mask_tensor,
                    return_q_values=True
                )
            else:
                action_logits, pref_values = self.grow_module(
                    state,
                    candidate_embeddings_tensor,
                    action_mask_tensor,
                    return_q_values=False
                )
            
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            if return_log_probs:
                log_probs.append(log_prob)
            
            if return_q_values:
                selected_q = q_vals[action]
                q_values.append(selected_q)
            
            action_item = action.item()
            
            if action_item == len(candidate_indices):  # STOP
                if step == 0:
                    stopped_immediately = True
                break
            
            selected_node = candidate_indices[action_item]
            corridor_prefix.append(selected_node)
            
            if return_pref_tensors:
                selected_pref = pref_values[action_item]
                selected_pref_tensors.append(selected_pref)
                pref_val = float(selected_pref.detach().item())
            else:
                pref_val = float(pref_values[action_item].detach().item())

            lane_ids = self.direction_to_lanes.get(selected_node, [])
            if lane_ids:
                for lane_id in lane_ids:
                    if lane_id not in preference_values:
                        preference_values[lane_id] = pref_val
            
            neighbor_junction = self.node_to_junction.get(selected_node)
            if neighbor_junction:
                used_junctions_in_corridor.add(neighbor_junction)
        
        seed_value = self._compute_seed_value(corridor_prefix)

        # Normalize lane-level preferences to sum to 1 (fallback to uniform).
        try:
            if isinstance(preference_values, dict) and len(preference_values) > 0:
                keys = list(preference_values.keys())
                vals = []
                for k in keys:
                    try:
                        v = float(preference_values.get(k, 0.0))
                    except Exception:
                        v = 0.0
                    if not torch.isfinite(torch.tensor(v)).item():
                        v = 0.0
                    if v < 0.0:
                        v = 0.0
                    vals.append(v)
                s = float(sum(vals))
                if s <= 0.0:
                    uni = 1.0 / float(len(keys)) if len(keys) > 0 else 0.0
                    for k in keys:
                        preference_values[k] = float(uni)
                else:
                    for k, v in zip(keys, vals):
                        preference_values[k] = float(v) / s
        except Exception:
            pass
        
        corridor = Corridor(
            direction_nodes=corridor_prefix,
            seed_value=seed_value,
            preference_values=preference_values,
            junctions=used_junctions_in_corridor,
            stopped_immediately=stopped_immediately
        )
        
        if return_log_probs and return_q_values and return_pref_tensors:
            return corridor, log_probs, q_values, selected_pref_tensors
        if return_log_probs and return_q_values:
            return corridor, log_probs, q_values
        if return_log_probs and return_pref_tensors:
            return corridor, log_probs, selected_pref_tensors
        if return_q_values and return_pref_tensors:
            return corridor, q_values, selected_pref_tensors
        if return_log_probs:
            return corridor, log_probs
        if return_q_values:
            return corridor, q_values
        if return_pref_tensors:
            return corridor, selected_pref_tensors
        return corridor
    
    def _get_neighbors(self, node_idx: int) -> List[int]:
        """Get a node's neighbors (from the direction graph)."""
        if self.direction_graph is None or self.direction_graph.edge_index is None:
            return []
        
        edge_index = self.direction_graph.edge_index
        neighbors = []
        
        for i in range(edge_index.shape[1]):
            if edge_index[0, i].item() == node_idx:
                neighbor = edge_index[1, i].item()
                neighbors.append(neighbor)
        
        return neighbors
    
    def _is_valid_action(
        self,
        corridor_prefix: List[int],
        candidate_node: int,
        used_junctions: Set[str]
    ) -> bool:
        """Return True if candidate is feasible."""
        if candidate_node in corridor_prefix:
            return False
        
        candidate_junction = self.node_to_junction.get(candidate_node)
        if candidate_junction and candidate_junction in used_junctions:
            return False
        
        return True
    
    def _compute_seed_value(self, corridor_nodes: List[int]) -> float:
        """
        Compute seed value (corridor reward).

        NOTE: this value is used only for logging/diagnostics (not for planning or training targets).
        """
        if self.direction_graph is None or getattr(self.direction_graph, "x", None) is None:
            return 0.0
        
        if len(corridor_nodes) == 0:
            return 0.0
        
        pressures = self.direction_graph.x
        seed_value = 0.0
        
        for i, node_idx in enumerate(corridor_nodes):
            if node_idx >= pressures.shape[0]:
                continue
            r_pressure = float(pressures[node_idx][0].item())
            seed_value += r_pressure
        
        return seed_value

