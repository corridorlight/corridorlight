"""
Traffic pressure graph builder.
"""

import os
import sys
from typing import Dict
from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import Data

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

import sumolib


class PressureGraphBuilder:
    """Build a direction-level graph and per-node pressure features."""
    
    def __init__(self, reward_fn: str = "diff-waiting-time", net_file: str = None):
        """Create a graph builder."""
        self.reward_fn = reward_fn
        self.net_file = net_file
        self.direction_graph = None  # direction graph (aggregated graph)
        self.lane_to_direction = {}  # lane id -> direction node id mapping
        self.direction_to_lanes = {}  # direction node id -> list of lane ids
        self.direction_to_junction = {}  # direction node id -> junction id
        self.edge_index = None  # PyG edge_index
        self.num_directions = 0

        self._last_lane_waiting_time: Dict[tuple, float] = {}
        self._lane_index_cache: Dict[str, Dict[str, int]] = {}
        
        if net_file:
            self.build_direction_graph(net_file)
        
    def build_direction_graph(self, net_file: str):
        """Build a direction graph from a SUMO net."""
        net = sumolib.net.readNet(net_file)
        
        lanes = {}
        lane_connections = defaultdict(list)  # lane_id -> [outgoing_lane_ids]
        
        for edge in net.getEdges():
            for lane in edge.getLanes():
                lane_id = lane.getID()
                lanes[lane_id] = lane
                
                for connection in lane.getOutgoing():
                    to_lane = connection.getToLane()
                    if to_lane:
                        lane_connections[lane_id].append(to_lane.getID())
        
        direction_groups = defaultdict(list)  # (from_junction, to_junction, direction) -> [lane_ids]
        
        for lane_id, lane in lanes.items():
            edge = lane.getEdge()
            from_junction = edge.getFromNode().getID() if edge.getFromNode() else None
            to_junction = edge.getToNode().getID() if edge.getToNode() else None
            
            direction_key = (from_junction, to_junction, lane.getIndex())
            direction_groups[direction_key].append(lane_id)
        
        self.lane_to_direction = {}
        self.direction_to_lanes = {}
        self.direction_to_junction = {}
        
        direction_idx = 0
        for direction_key, lane_ids in direction_groups.items():
            from_junction, to_junction, _ = direction_key
            
            self.direction_to_lanes[direction_idx] = lane_ids
            self.direction_to_junction[direction_idx] = to_junction
            
            for lane_id in lane_ids:
                self.lane_to_direction[lane_id] = direction_idx
            
            direction_idx += 1
        
        self.num_directions = direction_idx
        
        direction_edges = []
        for direction_idx in range(self.num_directions):
            lane_ids = self.direction_to_lanes[direction_idx]
            
            outgoing_directions = set()
            for lane_id in lane_ids:
                for out_lane_id in lane_connections.get(lane_id, []):
                    if out_lane_id in self.lane_to_direction:
                        out_direction = self.lane_to_direction[out_lane_id]
                        if out_direction != direction_idx:  # Avoid self-loops
                            outgoing_directions.add(out_direction)
            
            for out_direction in outgoing_directions:
                direction_edges.append([direction_idx, out_direction])
        
        if direction_edges:
            self.edge_index = torch.tensor(direction_edges, dtype=torch.long).t().contiguous()
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    def compute_pressure(self, lane_id: str, traffic_signal) -> float:
        """Compute lane-level proxy consistent with reward_fn."""
        ts_id = getattr(traffic_signal, "id", None)
        if ts_id is None:
            return 0.0
        ts_id = str(ts_id)

        lane_to_idx = self._lane_index_cache.get(ts_id)
        if lane_to_idx is None:
            lanes = list(getattr(traffic_signal, "lanes", []) or [])
            lane_to_idx = {lid: i for i, lid in enumerate(lanes)}
            self._lane_index_cache[ts_id] = lane_to_idx

        idx = lane_to_idx.get(lane_id, None)
        if idx is None:
            return 0.0
        if self.reward_fn == "diff-waiting-time":
            try:
                per_lane_wait = traffic_signal.get_accumulated_waiting_time_per_lane()
                lane_wait = float(per_lane_wait[idx]) / 100.0
            except Exception:
                lane_wait = 0.0

            key = (ts_id, lane_id)
            last = float(self._last_lane_waiting_time.get(key, lane_wait))
            reward = last - lane_wait
            self._last_lane_waiting_time[key] = lane_wait
            return float(reward)

        if self.reward_fn == "queue":
            try:
                q = traffic_signal.get_lanes_queue()
                return -float(q[idx]) / 50.0
            except Exception:
                return 0.0

        if self.reward_fn == "pressure":
            try:
                d = traffic_signal.get_lanes_density()
                return -float(d[idx])
            except Exception:
                return 0.0

        if self.reward_fn == "average-speed":
            try:
                mean_speed = float(traffic_signal.sumo.lane.getLastStepMeanSpeed(lane_id))
                allowed_speed = float(traffic_signal.sumo.lane.getAllowedSpeed(lane_id))
                if allowed_speed <= 0.0:
                    return 0.0
                if mean_speed < 0.0:
                    mean_speed = allowed_speed
                return float(mean_speed / allowed_speed)
            except Exception:
                return 0.0

        try:
            per_lane_wait = traffic_signal.get_accumulated_waiting_time_per_lane()
            lane_wait = float(per_lane_wait[idx]) / 100.0
            return -lane_wait
        except Exception:
            return 0.0
    
    def save_temporal_graph(self, timestep: int, pressure_dict: Dict[str, float]) -> Data:
        """Create a PyG Data graph with updated node features."""
        node_features = []
        
        for direction_idx in range(self.num_directions):
            lane_ids = self.direction_to_lanes[direction_idx]
            
            direction_pressures = []
            for lane_id in lane_ids:
                if lane_id in pressure_dict:
                    direction_pressures.append(pressure_dict[lane_id])
            
            if direction_pressures:
                avg_pressure = np.mean(direction_pressures)
            else:
                avg_pressure = 0.0
            
            node_features.append([avg_pressure])
        
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        
        data = Data(
            x=node_features_tensor,
            edge_index=self.edge_index,
            num_nodes=self.num_directions
        )
        
        return data

