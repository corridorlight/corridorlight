"""
Full Corridor RL trainer.
"""

import os
import sys
import time
import json
import shutil
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

import sumo_rl

from ..models.pressure_graph import PressureGraphBuilder
from ..models.gnn_encoder import GNNEncoder
from ..models.seed_module import SeedModule
from ..models.grow_module import GrowModule
from ..agents.corridor_agent import CorridorAgent
from ..agents.intersection_agent import IntersectionAgent
from .mgda_trainer import MGDATrainer


class CorridorRLTrainer:
    """
    Full Corridor RL trainer.

    Integrates the upper-level CorridorAgent and the lower-level IntersectionAgent.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict.
        """
        self.config = config

        # Deprecation: corridor_decayed mode is disabled (paper uses cooperative reward shaping: r^{coop}=r^{loc}+bonus).
        # Keep backward compatibility by auto-switching to shaping.
        try:
            _mode = str(self.config.get("reward_glob_mode", "shaping")).lower()
        except Exception:
            _mode = "shaping"
        if _mode == "corridor_decayed":
            print("Info: reward_glob_mode=corridor_decayed is deprecated; switching to shaping.")
            self.config["reward_glob_mode"] = "shaping"
            # Ensure shaping is effective by default.
            if float(self.config.get("corridor_global_reward_coef", 0.0)) == 0.0:
                self.config["corridor_global_reward_coef"] = 1.0
        
        device_str = config.get("device", "cpu")
        if device_str == "auto":
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            import torch
            self.device = torch.device(device_str)
        
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA: {torch.version.cuda}")
        
        self.net_file = config.get("net_file", "sumo_rl/nets/4x4-Lucas/4x4.net.xml")
        self.route_file = config.get("route_file", "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml")
        self.delta_time = config.get("delta_time", 5)
        self.corridor_delta_time = config.get("corridor_delta_time", 20)  # Upper-level decision interval
        
        self.env = None
        self.graph_builder = None
        self.gnn_encoder = None
        self.corridor_agent = None
        self.intersection_agents = {}  # agent_id -> IntersectionAgent
        self.mgda_trainers = {}  # agent_id -> MGDATrainer
        
        self.thread_pool = None
        self.num_threads = 1  # Default; updated in initialize()
        
        self.current_corridors = []  # Currently generated corridors
        self.last_corridor_update_time = 0
        
        self.skip_corridor_updates = self.config.get("skip_corridor_updates", False)
        
        self.disable_corridor_agent = self.config.get("disable_corridor_agent", False)
        
        self.disable_gnn = self.config.get("disable_gnn", False)
        if self.disable_gnn:
            self.disable_corridor_agent = True
            print("Info: GNN disabled; Corridor Agent disabled as well.")

        collect_lane_features = self.config.get("collect_lane_features", None)
        if collect_lane_features is None:
            self.collect_lane_features = not self.disable_corridor_agent
        else:
            self.collect_lane_features = bool(collect_lane_features)
        
        self.episode_start_times = {}
        self.episode_durations = deque(maxlen=100)
        self.sumo_episode_offset = 0
        
        self.save_dir = Path(self.config.get("save_dir", "models"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tripinfo_file = str(self.save_dir / "tripinfo.xml")
        
        self.checkpoint_interval = int(self.config.get("checkpoint_interval", 0))
        self.checkpoint_save_dir = Path(self.config.get("checkpoint_save_dir", "checkpoints"))
        self.checkpoint_save_models = bool(self.config.get("checkpoint_save_models", True))
        self.checkpoint_save_training_state = bool(self.config.get("checkpoint_save_training_state", True))
        self.checkpoint_method_name = self.config.get("agent_type", "corridor")
        net_file = self.config.get("net_file", "")
        if net_file:
            net_path = Path(net_file)
            self.checkpoint_map_name = net_path.stem.replace(".net", "").replace("_net", "")
        else:
            self.checkpoint_map_name = "map"
        
        self.logger = None
        
        self.training_callback = None
        
        self.training_metrics_history = []

        self._episode_corridor_logs = []
        self._episode_lambda_logs = []
        self._episode_preference_logs = []
        self._episode_reward_logs = []
        
        self._last_lane_waiting_time = {}
        
    def initialize(self):
        """Initialize all components."""
        try:
            project_root = Path(__file__).resolve().parents[2]
        except Exception:
            project_root = Path.cwd()

        def _resolve_sumo_path(p: str) -> str:
            try:
                pp = Path(str(p))
                if pp.is_absolute():
                    return str(pp)
                cand = (project_root / pp).resolve()
                return str(cand) if cand.exists() else str(pp)
            except Exception:
                return str(p)

        def _split_csv_paths(s: str):
            return [x.strip() for x in str(s).split(",") if str(x).strip()]

        def _parse_sumocfg(sumocfg_path: str) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            try:
                root = ET.parse(sumocfg_path).getroot()
            except Exception:
                return result

            def _get_child(parent, name: str):
                for child in parent:
                    if child.tag == name:
                        return child
                return None

            input_node = _get_child(root, "input")
            if input_node is not None:
                for item in input_node:
                    if item.tag == "net-file":
                        result["net_file"] = item.attrib.get("value", "")
                    elif item.tag == "route-files":
                        result["route_file"] = item.attrib.get("value", "")
                    elif item.tag == "additional-files":
                        result["additional_files"] = item.attrib.get("value", "")

            time_node = _get_child(root, "time")
            if time_node is not None:
                begin = None
                end = None
                for item in time_node:
                    if item.tag == "begin":
                        try:
                            begin = float(item.attrib.get("value", "0"))
                        except Exception:
                            begin = None
                    elif item.tag == "end":
                        try:
                            end = float(item.attrib.get("value", "0"))
                        except Exception:
                            end = None
                if begin is not None and end is not None and end > begin:
                    result["num_seconds"] = int(end - begin)

            extra_cmd_parts = []
            processing_node = _get_child(root, "processing")
            if processing_node is not None:
                for item in processing_node:
                    if item.tag == "ignore-route-errors":
                        extra_cmd_parts.append(f"--ignore-route-errors {item.attrib.get('value','true')}")

            report_node = _get_child(root, "report")
            if report_node is not None:
                for item in report_node:
                    if item.tag == "no-step-log":
                        extra_cmd_parts.append(f"--no-step-log {item.attrib.get('value','true')}")
                    elif item.tag == "duration-log.statistics":
                        extra_cmd_parts.append(f"--duration-log.statistics {item.attrib.get('value','true')}")
                    elif item.tag == "verbose":
                        extra_cmd_parts.append(f"--verbose {item.attrib.get('value','true')}")

            routing_node = _get_child(root, "routing")
            if routing_node is not None:
                for item in routing_node:
                    if item.tag == "device.rerouting.adaptation-steps":
                        extra_cmd_parts.append(f"--device.rerouting.adaptation-steps {item.attrib.get('value','0')}")
                    elif item.tag == "device.rerouting.adaptation-interval":
                        extra_cmd_parts.append(f"--device.rerouting.adaptation-interval {item.attrib.get('value','0')}")

            for item in root:
                if item.tag == "time-to-teleport":
                    extra_cmd_parts.append(f"--time-to-teleport {item.attrib.get('value','-1')}")

            if extra_cmd_parts:
                result["additional_sumo_cmd"] = " ".join(extra_cmd_parts)
            return result

        sumocfg_file = self.config.get("sumocfg_file", "") or ""
        sumocfg_data: Dict[str, Any] = {}
        if str(sumocfg_file).strip():
            sumocfg_path = _resolve_sumo_path(sumocfg_file)
            if Path(sumocfg_path).exists():
                sumocfg_data = _parse_sumocfg(sumocfg_path)
                if sumocfg_data.get("net_file"):
                    sumocfg_data["net_file"] = _resolve_sumo_path(Path(sumocfg_path).parent / sumocfg_data["net_file"])
                if sumocfg_data.get("route_file"):
                    routes = _split_csv_paths(sumocfg_data["route_file"])
                    routes = [_resolve_sumo_path(Path(sumocfg_path).parent / r) for r in routes]
                    sumocfg_data["route_file"] = ",".join(routes)
                if sumocfg_data.get("additional_files"):
                    adds = _split_csv_paths(sumocfg_data["additional_files"])
                    adds = [_resolve_sumo_path(Path(sumocfg_path).parent / a) for a in adds]
                    sumocfg_data["additional_files"] = ",".join(adds)

        self.net_file = _resolve_sumo_path(self.net_file)
        route_parts = _split_csv_paths(self.route_file)
        if not route_parts:
            if sumocfg_data.get("route_file"):
                route_parts = _split_csv_paths(sumocfg_data["route_file"])
            else:
                route_parts = [str(self.route_file)]
        route_parts = [_resolve_sumo_path(p) for p in route_parts]
        self.route_file = ",".join(route_parts)

        additional_files = self.config.get("additional_files", "") or ""
        add_parts = _split_csv_paths(additional_files)
        if not add_parts and sumocfg_data.get("additional_files"):
            add_parts = _split_csv_paths(sumocfg_data["additional_files"])
        add_parts = [_resolve_sumo_path(p) for p in add_parts] if add_parts else []

        if sumocfg_data.get("num_seconds") and not self.config.get("num_seconds"):
            self.config["num_seconds"] = sumocfg_data["num_seconds"]
        if not Path(self.net_file).exists():
            raise FileNotFoundError(f"net_file not found (check working directory/path): {self.net_file}")
        missing_routes = [p for p in route_parts if not Path(p).exists()]
        if missing_routes:
            raise FileNotFoundError(f"route_file not found (check working directory/path): {missing_routes}")
        missing_add = [p for p in add_parts if not Path(p).exists()]
        if missing_add:
            raise FileNotFoundError(f"additional_files not found (check working directory/path): {missing_add}")

        additional_cmd = f"--tripinfo-output {self.tripinfo_file}"
        traffic_scale = float(self.config.get("traffic_scale", 1.0))
        if traffic_scale != 1.0:
            additional_cmd = f"{additional_cmd} --scale {traffic_scale}"

        if add_parts:
            additional_cmd = f"{additional_cmd} --additional-files {','.join(add_parts)}"
        extra_cmd = self.config.get("additional_sumo_cmd", "") or ""
        sumocfg_extra_cmd = sumocfg_data.get("additional_sumo_cmd", "")
        if str(sumocfg_extra_cmd).strip():
            additional_cmd = f"{additional_cmd} {str(sumocfg_extra_cmd).strip()}"
        if str(extra_cmd).strip():
            additional_cmd = f"{additional_cmd} {str(extra_cmd).strip()}"
        add_system_info = self.config.get("add_system_info", False)  # Disabled by default
        add_per_agent_info = self.config.get("add_per_agent_info", False)  # Disabled by default
        
        self.env = sumo_rl.parallel_env(
            net_file=self.net_file,
            route_file=self.route_file,
            use_gui=self.config.get("gui", False),
            num_seconds=self.config.get("num_seconds", 8000),
            min_green=self.config.get("min_green", 5),
            delta_time=self.delta_time,
            reward_fn=self.config.get("reward_fn", "diff-waiting-time"),
            reward_scale=float(self.config.get("reward_scale", 1.0)),
            add_system_info=add_system_info,
            add_per_agent_info=add_per_agent_info,
            additional_sumo_cmd=additional_cmd,
        )

        try:
            base_env = self._get_base_env()
            if base_env is not None:
                setattr(base_env, "congestion_force_green", bool(self.config.get("congestion_force_green", False)))
                setattr(base_env, "congestion_metric", str(self.config.get("congestion_metric", "queue")))
                setattr(base_env, "congestion_queue_threshold", float(self.config.get("congestion_queue_threshold", 0.9)))
                setattr(base_env, "congestion_density_threshold", float(self.config.get("congestion_density_threshold", 0.9)))
        except Exception:
            pass
        
        self.num_threads = max(1, int(self.config.get("num_threads", 1)))
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads) if self.num_threads > 1 else None
        
        if not self.disable_corridor_agent and not self.disable_gnn:
            self.graph_builder = PressureGraphBuilder(
                reward_fn=self.config.get("reward_fn", "diff-waiting-time"),
                net_file=self.net_file
            )
            
            input_dim = 1  # Pressure value dimension
            hidden_dim = self.config.get("gnn_hidden_dim", 128)
            output_dim = self.config.get("gnn_output_dim", 64)
            num_layers = self.config.get("gnn_num_layers", 2)
            architecture = self.config.get("gnn_architecture", "GCN")
            
            self.gnn_encoder = GNNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                architecture=architecture
            ).to(self.device)
            
            seed_module = SeedModule(
                embedding_dim=output_dim,
                hidden_dim=self.config.get("seed_projection_dim", 64)
            ).to(self.device)
            
            grow_module = GrowModule(
                embedding_dim=output_dim,
                hidden_dim=hidden_dim,
                max_neighbors=4,
                max_length=self.config.get("max_corridor_length", 10)
            ).to(self.device)
            
            self.corridor_agent = CorridorAgent(
                gnn_encoder=self.gnn_encoder,
                seed_module=seed_module,
                grow_module=grow_module,
                config=self.config
            )
            
            pressure_graph = self.graph_builder.save_temporal_graph(0, {})
            self.corridor_agent.set_direction_graph(pressure_graph, self.graph_builder)
            
            ppo_lr = self.config.get("ppo_lr", 3e-4)
            self.gnn_optimizer = torch.optim.Adam(
                self.gnn_encoder.parameters(),
                lr=ppo_lr
            )
            self.seed_module_optimizer = torch.optim.Adam(
                self.corridor_agent.seed_module.parameters(),
                lr=ppo_lr
            )
            self.grow_module_optimizer = torch.optim.Adam(
                self.corridor_agent.grow_module.parameters(),
                lr=ppo_lr
            )
            print("Corridor Agent optimizers created (GNN/Seed/Grow).")
        else:
            self.graph_builder = None
            self.gnn_encoder = None
            self.corridor_agent = None
            self.gnn_optimizer = None
            self.seed_module_optimizer = None
            self.grow_module_optimizer = None
            if self.disable_gnn:
                print("Info: GNN disabled; using standard observations.")
            else:
                print("Info: Corridor Agent disabled; single-objective mode.")
        
        obs_dict = self.env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        
        self.agent_ids = list(self.env.agents) if hasattr(self.env, 'agents') else list(obs_dict.keys())
        
        base_env = self._get_base_env()
        share_flag = bool(self.config.get("share_intersection_parameters", False))
        try:
            net_s = f"{self.net_file} {self.route_file}"
            is_4x4 = ("4x4" in net_s.lower())
        except Exception:
            is_4x4 = False
        share_flag = bool(share_flag and is_4x4)

        per_agent_lane_ids: Dict[str, list] = {}
        obs_dims = set()
        action_dims = set()
        max_lanes = 0
        for agent_id in self.agent_ids:
            obs = obs_dict.get(agent_id, np.zeros(20))
            try:
                obs_dims.add(int(len(obs)))
            except Exception:
                obs_dims.add(20)
            try:
                action_dims.add(int(self.env.action_space(agent_id).n))
            except Exception:
                action_dims.add(0)
            lane_ids = []
            try:
                ts = base_env.traffic_signals.get(agent_id) if hasattr(base_env, "traffic_signals") else None
                lane_ids = list(getattr(ts, "lanes", []) or []) if ts is not None else []
            except Exception:
                lane_ids = []
            per_agent_lane_ids[agent_id] = lane_ids
            if len(lane_ids) > max_lanes:
                max_lanes = len(lane_ids)

        if share_flag and (len(obs_dims) == 1) and (len(action_dims) == 1) and (max_lanes > 0):
            lane_feature_dim = 10
            shared_obs_dim = int(list(obs_dims)[0])
            shared_action_dim = int(list(action_dims)[0])
            shared_task_dim = int(2 * max_lanes)

            agent0_id = self.agent_ids[0]
            shared_agent0 = IntersectionAgent(
                observation_dim=shared_obs_dim,
                action_dim=shared_action_dim,
                lane_feature_dim=lane_feature_dim,
                config=self.config,
                lane_ids=per_agent_lane_ids.get(agent0_id, []),
                corridor_task_dim=shared_task_dim,
                shared_modules=None
            )
            shared_modules = {
                "backbone": shared_agent0.backbone,
                "local_head": shared_agent0.local_head,
                "global_head": shared_agent0.global_head,
                "value_loc_head": shared_agent0.value_loc_head,
                "value_glob_head": shared_agent0.value_glob_head,
                "delta_head": getattr(shared_agent0, "delta_head", None),
            }
            ppo_lr = float(self.config.get("ppo_lr", 3e-4))
            shared_optimizers = {
                "backbone": torch.optim.Adam(shared_agent0.backbone.parameters(), lr=ppo_lr),
                "local_head": torch.optim.Adam(shared_agent0.local_head.parameters(), lr=ppo_lr),
                "global_head": torch.optim.Adam(shared_agent0.global_head.parameters(), lr=ppo_lr),
                "value_loc": torch.optim.Adam(shared_agent0.value_loc_head.parameters(), lr=ppo_lr),
                "value_glob": torch.optim.Adam(shared_agent0.value_glob_head.parameters(), lr=ppo_lr),
            }
            # Optional: gate-delta head (used when aligning gating with advantage-based rule).
            if getattr(shared_agent0, "delta_head", None) is not None:
                shared_optimizers["delta_head"] = torch.optim.Adam(shared_agent0.delta_head.parameters(), lr=ppo_lr)

            self.intersection_agents[agent0_id] = shared_agent0
            self.mgda_trainers[agent0_id] = MGDATrainer(shared_agent0, self.config, optimizers=shared_optimizers)

            for agent_id in self.agent_ids[1:]:
                obs = obs_dict.get(agent_id, np.zeros(shared_obs_dim))
                intersection_agent = IntersectionAgent(
                    observation_dim=len(obs) if hasattr(obs, "__len__") else shared_obs_dim,
                    action_dim=shared_action_dim,
                    lane_feature_dim=lane_feature_dim,
                    config=self.config,
                    lane_ids=per_agent_lane_ids.get(agent_id, []),
                    corridor_task_dim=shared_task_dim,
                    shared_modules=shared_modules
                )
                self.intersection_agents[agent_id] = intersection_agent
                self.mgda_trainers[agent_id] = MGDATrainer(intersection_agent, self.config, optimizers=shared_optimizers)

            print(f"4x4 parameter sharing enabled: obs_dim={shared_obs_dim}, action_dim={shared_action_dim}, task_dim={shared_task_dim}")
        else:
            if share_flag:
                print("Warning: --share requested but cannot share (obs/action dims mismatch or no lanes). Falling back to per-intersection params.")
            for agent_id in self.agent_ids:
                # Use the standard sumo_rl observation (consistent with MoLLMLight), not GNN encoding
                obs = obs_dict.get(agent_id, np.zeros(20))  # Default observation dimension
                action_dim = self.env.action_space(agent_id).n
                lane_feature_dim = 10
                lane_ids = per_agent_lane_ids.get(agent_id, [])
                intersection_agent = IntersectionAgent(
                    observation_dim=len(obs),
                    action_dim=action_dim,
                    lane_feature_dim=lane_feature_dim,
                    config=self.config,
                    lane_ids=lane_ids
                )
                self.intersection_agents[agent_id] = intersection_agent
                self.mgda_trainers[agent_id] = MGDATrainer(intersection_agent, self.config)
        
        # Performance optimization: pre-initialize lane_features cache
        self._lane_features_cache = {}
        if self.collect_lane_features:
            for agent_id in self.agent_ids:
                self._get_lane_features(agent_id)  # Pre-create cache
        
        # Initialize wandb (if enabled)
        self.logger = self._setup_logging()
    
    def _get_base_env(self):
        """Try to unwrap wrappers to get the underlying SUMO-RL environment."""
        env = self.env
        if hasattr(env, "unwrapped"):
            try:
                env = env.unwrapped
            except Exception:
                pass
        if hasattr(env, "env"):
            try:
                env = env.env
            except Exception:
                pass
        return env

    def _select_actions(self, obs_dict: Dict[str, np.ndarray], training: bool = True, return_infos: bool = False):
        """
        Select actions (supports multithreading; optimized sync-update version).

        Important: the lower-level IntersectionAgent uses the standard sumo_rl observation (consistent with
        MoLLMLight), not GNN encodings. GNN encodings are only used by the upper-level CorridorAgent to
        generate corridors.
        
        Args:
            obs_dict: Observation dict (standard sumo_rl observation, consistent with MoLLMLight).
            training: Whether in training mode.
            return_infos: Whether to return per-agent info dicts (to avoid recomputation during training).
            
        Returns:
            If return_infos=False: Dict[str, int] (actions)
            If return_infos=True: Tuple[Dict[str, int], Dict[str, Dict[str, Any]]] (actions and infos)
        """
        actions: Dict[str, int] = {}
        infos: Dict[str, Dict[str, Any]] = {}
        
        # Optimization: batch-fetch all lane_features (reduce function call overhead)
        lane_features_dict = {}
        if self.collect_lane_features:
            for agent_id in self.agent_ids:
                if agent_id in obs_dict:
                    lane_features_dict[agent_id] = self._get_lane_features(agent_id)
        else:
            for agent_id in self.agent_ids:
                if agent_id in obs_dict:
                    lane_features_dict[agent_id] = {}

        # If reward_glob_mode is same_as_loc, force single-objective behavior (avoid global head interference)
        reward_glob_mode = str(self.config.get("reward_glob_mode", "shaping")).lower()
        force_local_only_all = (reward_glob_mode == "same_as_loc")

        # Whether this step belongs to any corridor: if lane_mask is empty/all False, force single-objective
        # (do not use the global head).
        force_local_only_map: Dict[str, bool] = {}
        if not self.disable_corridor_agent:
            for agent_id in self.agent_ids:
                agent = self.intersection_agents.get(agent_id)
                if agent is None:
                    force_local_only_map[agent_id] = True
                    continue
                try:
                    lane_mask = getattr(agent, "lane_mask", {}) or {}
                    in_corridor = any(bool(v) for v in lane_mask.values()) if lane_mask else False
                except Exception:
                    in_corridor = False
                force_local_only_map[agent_id] = force_local_only_all or (not in_corridor)
        else:
            for agent_id in self.agent_ids:
                force_local_only_map[agent_id] = True
        
        if hasattr(self, 'thread_pool') and self.thread_pool and len(self.agent_ids) > 1:
            # Select actions in parallel using a thread pool (batched)
            futures = {}
            for agent_id in self.agent_ids:
                if agent_id in obs_dict and agent_id in lane_features_dict:
                    # Use standard sumo_rl observation (consistent with MoLLMLight), not GNN encoding
                    # Device conversion is handled inside select_action; no need to do it here.
                    obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32)
                    lane_features = lane_features_dict[agent_id]
                    futures[self.thread_pool.submit(
                        self.intersection_agents[agent_id].select_action,
                        obs, lane_features, training, return_infos, force_local_only_map.get(agent_id, True)
                    )] = agent_id
            
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    result = future.result()
                    if return_infos:
                        action, info = result
                        actions[agent_id] = action
                        infos[agent_id] = info
                    else:
                        action, _ = result
                        actions[agent_id] = action
                except Exception as e:
                    print(f"Warning: action selection failed ({agent_id}): {e}")
                    actions[agent_id] = 0
        else:
            # Select actions sequentially (batched; reduce repeated computation)
            for agent_id in self.agent_ids:
                if agent_id in obs_dict and agent_id in lane_features_dict:
                    # Device conversion is handled inside select_action
                    obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32)
                    lane_features = lane_features_dict[agent_id]
                    try:
                        result = self.intersection_agents[agent_id].select_action(
                            obs, lane_features, training, return_infos, force_local_only_map.get(agent_id, True)
                        )
                        if return_infos:
                            action, info = result
                            actions[agent_id] = action
                            infos[agent_id] = info
                        else:
                            action, _ = result
                            actions[agent_id] = action
                    except Exception as e:
                        # Avoid a single agent's NaN/numerical issue stopping the whole training
                        print(f"Warning: action selection failed ({agent_id}): {e}")
                        actions[agent_id] = 0
                        if return_infos:
                            infos[agent_id] = {"lambda": 0.0, "mode": "action_selection_failed"}
        
        if return_infos:
            return actions, infos
        return actions
    
    def train(self, num_episodes: int = 1000):
        """Main training loop (optimized: synchronous updates with full training logic)."""
        print(f"Training started: {num_episodes} episode(s)")
        sys.stdout.flush()
        
        self.sumo_episode_offset = 1
        
        # Initialize Corridor Agent update counter
        if not self.disable_corridor_agent and not self.disable_gnn:
            self._corridor_update_count = 0
        
        for episode in range(num_episodes):
            self.start_episode_timing(episode + 1)
            
            if episode == 0:
                print(f"Episode {episode + 1}/{num_episodes}...")
                sys.stdout.flush()
            
            rollout_data = self._collect_episode_data()
            
            if rollout_data:
                self._update_agents(rollout_data)
            
            episode_duration = self.end_episode_timing(episode + 1)
            
            travel_time_metrics = self._parse_tripinfo_avg_travel_time(episode + 1)
            
            avg_reward = rollout_data.get('avg_reward', 0.0) if rollout_data else 0.0
            episode_length = rollout_data.get('episode_length', 0) if rollout_data else 0
            episode_rewards = rollout_data.get('episode_rewards', {}) if rollout_data else {}
            avg_queue_length = rollout_data.get('avg_queue_length', 0.0) if rollout_data else 0.0
            avg_speed_mps = rollout_data.get('avg_speed_mps', 0.0) if rollout_data else 0.0
            avg_speed_norm = rollout_data.get('avg_speed_norm', 1.0) if rollout_data else 1.0
            
            metrics = {
                'avg_reward': avg_reward,
                'episode_rewards': episode_rewards,
                'episode_duration': episode_duration,
                'avg_queue_length': float(avg_queue_length),
                'avg_speed_mps': float(avg_speed_mps),
                'avg_speed_norm': float(avg_speed_norm),
            }
            
            if travel_time_metrics:
                metrics.update(travel_time_metrics)
                if "avg_speed_mps_tripinfo" in travel_time_metrics:
                    avg_speed_mps = float(travel_time_metrics.get("avg_speed_mps_tripinfo", avg_speed_mps))
                    metrics["avg_speed_mps"] = float(avg_speed_mps)
                trip_count = float(travel_time_metrics.get("tripinfo_count", 0.0))
                if trip_count > 0:
                    metrics["throughput_count"] = float(trip_count)
                    if episode_length > 0:
                        metrics["throughput_per_step"] = float(trip_count) / float(episode_length)
                        sim_seconds = float(episode_length) * float(getattr(self, "delta_time", 1.0))
                        if sim_seconds > 0:
                            metrics["throughput_per_hour"] = float(trip_count) / (sim_seconds / 3600.0)

            metrics.update(self._build_corridor_logging_metrics())

            # Log to wandb
            self._log_metrics(episode + 1, metrics)
            
            # Save training metrics history (for checkpointing)
            self.training_metrics_history.append(metrics)
            if len(self.training_metrics_history) > 100:
                self.training_metrics_history.pop(0)
            
            # Save checkpoint (if configured)
            if self._should_save_checkpoint(episode + 1):
                self._save_checkpoint(episode + 1)
            
            # Periodic evaluation (if configured)
            eval_interval = self.config.get("eval_interval", 0)
            if eval_interval > 0 and (episode + 1) % eval_interval == 0:
                eval_metrics = self._evaluate_agent(episode + 1)
                if eval_metrics:
                    # Add prefix for wandb
                    eval_metrics_for_log = {f"eval_{k}": v for k, v in eval_metrics.items()}
                    self._log_metrics(episode + 1, eval_metrics_for_log)
                    print(
                        f"Eval (episode {episode + 1}): "
                        f"avg_reward={eval_metrics.get('avg_reward', 0.0):.2f}, "
                        f"avg_travel_time={eval_metrics.get('avg_travel_time', 0.0):.2f}s, "
                        f"throughput_per_hour={eval_metrics.get('throughput_per_hour', 0.0):.2f}"
                    )
            
            # Invoke training monitor callback (if set)
            if self.training_callback:
                try:
                    # Get loss info and parameter-change flags from an MGDA trainer
                    training_metrics = metrics.copy()
                    param_changed = False
                    policy_loss = None
                    value_loss = None
                    
                    for agent_id in self.agent_ids:
                        if agent_id in self.mgda_trainers:
                            trainer = self.mgda_trainers[agent_id]
                            # Check whether parameters changed
                            if hasattr(trainer, '_last_backbone_params'):
                                import torch
                                for old_p, new_p in zip(trainer._last_backbone_params, self.intersection_agents[agent_id].backbone.parameters()):
                                    if not torch.allclose(old_p, new_p, atol=1e-6):
                                        param_changed = True
                                        break
                            # Get losses (if available)
                            if hasattr(trainer, '_last_policy_loss'):
                                policy_loss = trainer._last_policy_loss
                            if hasattr(trainer, '_last_value_loss'):
                                value_loss = trainer._last_value_loss
                            break
                    
                    training_metrics['param_changed'] = param_changed
                    if policy_loss is not None:
                        training_metrics['policy_loss'] = policy_loss
                    if value_loss is not None:
                        training_metrics['value_loss'] = value_loss
                    
                    self.training_callback(episode + 1, training_metrics)
                except Exception as e:
                    print(f"Warning: training callback error: {e}")
            
            # Progress line (no episode length / corridor details / timing breakdown)
            if travel_time_metrics and travel_time_metrics.get('avg_travel_time', 0) > 0:
                th_h = float(metrics.get("throughput_per_hour", 0.0))
                th_cnt = float(metrics.get("throughput_count", 0.0))
                print(
                    f"Episode {episode + 1}/{num_episodes}: "
                    f"avg_reward={avg_reward:.2f}, "
                    f"avg_travel_time={travel_time_metrics.get('avg_travel_time', 0):.2f}s, "
                    f"throughput_count={th_cnt:.0f}, "
                    f"throughput_per_hour={th_h:.2f}"
                )
            else:
                print(f"Episode {episode + 1}/{num_episodes}: avg_reward={avg_reward:.2f}")
            
            sys.stdout.flush()
        
        print("Training completed.")
        sys.stdout.flush()
    
    def _collect_episode_data(self) -> Dict[str, Any]:
        """Collect rollout data for one episode."""
        profile_timing = bool(self.config.get("profile_timing", False))
        timing = {
            "corridor_update": 0.0,
            "select_actions": 0.0,
            "env_step": 0.0,
            "post_step": 0.0,
            "steps": 0
        }
        obs_dict = self.env.reset()
        if isinstance(obs_dict, tuple):
            obs_dict = obs_dict[0]
        
        # Initialize a buffer for each agent
        buffers = {}
        for agent_id in self.agent_ids:
            buffers[agent_id] = {
                'states': [],
                'actions': [],
                'rewards': [],
                # Global-task reward in multi-objective mode (optional reward shaping; defaults to rewards)
                'rewards_glob': [],
                'values_loc': [],
                'values_glob': [],
                'log_probs': [],
                'lambdas': [],
                # Whether this step belongs to any corridor (mask global signal/training)
                'in_corridor': [],
                # Corridor task vector (mask+preference aligned to lane order), used to train global policy/critic
                'corridor_task_vec': [],
                'dones': [],
                'lane_features': [],
                # Optional: grow-Q observation for the global critic (recorded per step; aligned with lane_features)
                'corridor_q_observations': [],
            }
        
        episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        episode_length = 0
        
        # Reset corridor state
        self.current_corridors = []
        self.corridor_log_probs = []  # Training: log_probs generated by corridor policy
        self.corridor_q_values = []  # Training: Q-values produced by the corridor components
        self.corridor_v_values = []  # Training: seed V-values
        self.corridor_pref_tensors = []  # Training: selected preference tensors (keep gradients)
        self.corridor_update_steps = []  # Step indices of each corridor update
        step_count = 0
        
        # Initialize current episode step counter (align corridor update timing)
        self._current_episode_step = 0
        
        # reward_glob setting:
        # - local is the per-intersection aggregated reward (env per-agent reward)
        # - global is the (optional) cooperative reward stream (paper-aligned: shaping)
        reward_glob_mode = str(self.config.get("reward_glob_mode", "shaping")).lower()
        reward_fn = str(self.config.get("reward_fn", "diff-waiting-time")).lower()
        # Treat same_as_loc as "do not use corridor at runtime" to avoid corridor sampling consuming RNG
        # and changing behavior.
        use_corridor_runtime = (not self.disable_corridor_agent) and (not self.disable_gnn) and (reward_glob_mode != "same_as_loc")

        # Set training-mode flag (do not train upper level when same_as_loc)
        self._training_corridor_components = bool(use_corridor_runtime)

        # Reset episode logs
        self._reset_episode_logs()
        
        # Important: clear the lower-level corridor task at the start of each episode.
        # Otherwise, before the first corridor update, agents may reuse the previous episode's
        # lane_mask / preference / q, causing an agent that is not in any corridor to still use the global head.
        try:
            for _aid, _agent in (self.intersection_agents or {}).items():
                try:
                    _agent.set_corridor_task({}, {}, corridor_q_value=0.0)
                except Exception:
                    continue
        except Exception:
            pass

        # === Episode-level traffic metrics (avoid SUMO multi-run pollution: read-only; do not change state) ===
        # average queue length: average network-wide total queued (halting vehicles) per step
        # average speed: vehicle-step weighted; output m/s and normalized (speed/allowed_speed)
        base_env_for_metrics = self._get_base_env()
        queue_total_sum = 0.0
        queue_step_count = 0
        lane_count_by_agent: Dict[str, int] = {}
        try:
            ts_map = getattr(base_env_for_metrics, "traffic_signals", {}) if base_env_for_metrics is not None else {}
            if isinstance(ts_map, dict):
                for aid, ts in ts_map.items():
                    lanes = list(getattr(ts, "lanes", []) or [])
                    lane_count_by_agent[str(aid)] = max(1, len(lanes))
        except Exception:
            lane_count_by_agent = {}
        
        # Whether to use corridor at runtime (skip when same_as_loc to avoid RNG interference)
        use_corridor_runtime = (not self.disable_corridor_agent) and (not self.disable_gnn) and (reward_glob_mode != "same_as_loc")
        # Set training-mode flag (do not train upper level when same_as_loc)
        self._training_corridor_components = bool(use_corridor_runtime)

        # Compute corridor update interval (optimization: precompute)
        # If corridor is not used, skip all corridor-related computations.
        if use_corridor_runtime:
            corridor_update_interval = (
                max(1, self.corridor_delta_time // self.delta_time) 
                if not self.skip_corridor_updates 
                else float('inf')
            )
        else:
            corridor_update_interval = float('inf')

        # reward_glob settings have been initialized above

        # Legacy shaping (still available for compatibility)
        corridor_global_reward_coef = float(self.config.get("corridor_global_reward_coef", 0.0))
        corridor_global_reward_metric = str(self.config.get("corridor_global_reward_metric", "queue")).lower()
        reward_log_window = int(self.config.get("reward_log_window", 100))
        reward_log_flush_partial = bool(self.config.get("reward_log_flush_partial", True))
        reward_log_max_rows = int(self.config.get("reward_log_max_rows", 50000))
        if reward_log_window <= 0:
            reward_log_window = 100

        reward_window_acc = {
            aid: {"sum_loc": 0.0, "sum_glob": 0.0, "count": 0, "start_step": 0}
            for aid in self.agent_ids
        }

        reward_scale = float(self.config.get("reward_scale", 1.0))

        lane_diff_by_lane: Optional[Dict[str, float]] = None
        def _lane_metric(feat: Any, metric: str, lane_id: Optional[str] = None) -> float:
            """Extract lane-level reward from lane_features (larger = worse, so return negative values)."""
            try:
                if isinstance(feat, torch.Tensor):
                    feat_t = feat
                else:
                    feat_t = torch.tensor(feat, dtype=torch.float32)
                density = float(feat_t[0].item()) if feat_t.numel() > 0 else 0.0
                queue = float(feat_t[1].item()) if feat_t.numel() > 1 else 0.0
                wait = float(feat_t[2].item()) if feat_t.numel() > 2 else 0.0
                speed = float(feat_t[3].item()) if feat_t.numel() > 3 else 0.0
                if metric == "density":
                    v = density
                    val = -float(v)
                elif metric == "queue_density":
                    v = density + queue
                    val = -float(v)
                elif metric == "diff-waiting-time":
                    # Use per-lane waiting time difference as a proxy
                    if lane_id is None:
                        return 0.0
                    if isinstance(lane_diff_by_lane, dict) and lane_id in lane_diff_by_lane:
                        return float(lane_diff_by_lane[lane_id])
                    last = float(self._last_lane_waiting_time.get(lane_id, wait))
                    val = float(last - wait)  # Keep original scale (1.0)
                    self._last_lane_waiting_time[lane_id] = wait
                elif metric == "waiting-time-reduction":
                    if lane_id is None:
                        return 0.0
                    if isinstance(lane_diff_by_lane, dict) and lane_id in lane_diff_by_lane:
                        return float(lane_diff_by_lane[lane_id])
                    last = float(self._last_lane_waiting_time.get(lane_id, wait))
                    val = max(float(last - wait), 0.0)
                    self._last_lane_waiting_time[lane_id] = wait
                elif metric == "pressure":
                    # Proxy for lane-level pressure: negative density
                    val = -float(density)
                elif metric == "average-speed":
                    # Lane-level average speed (already normalized to 0~1)
                    val = float(speed)
                else:  # "queue"
                    v = queue
                    val = -float(v)
                return float(val) * reward_scale
            except Exception:
                return 0.0

        def _corridor_global_bonus(_agent_id: str, _lane_features: Dict[str, Any]) -> float:
            """Legacy: reward shaping using (preference-weighted) queue/density on corridor lanes.

            Returns a bonus (usually negative).
            """
            if corridor_global_reward_coef == 0.0:
                return 0.0
            agent_obj = self.intersection_agents.get(_agent_id)
            if agent_obj is None:
                return 0.0
            lane_mask = getattr(agent_obj, "lane_mask", {}) or {}
            pref = getattr(agent_obj, "preference_values", {}) or {}
            if not lane_mask:
                return 0.0
            num = 0.0
            den = 0.0
            for lane_id, feat in (_lane_features or {}).items():
                if lane_mask and not lane_mask.get(lane_id, False):
                    continue
                lane_r = _lane_metric(feat, corridor_global_reward_metric, lane_id)
                weight = 1.0 + float(pref.get(lane_id, 0.0))
                num += float(weight) * float(lane_r)
                den += float(weight)
            if den <= 0.0:
                return 0.0
            return float(num / den)
        
        while self.env.agents:
            step_count += 1
            
            if use_corridor_runtime:
                should_update_corridor = (
                    not self.skip_corridor_updates and 
                    step_count % corridor_update_interval == 0
                ) or (self.skip_corridor_updates and step_count == 1)
                
                if should_update_corridor:
                    self._current_corridor_update_step = episode_length
                    t0 = time.perf_counter()
                    self._update_corridors()
                    if profile_timing:
                        timing["corridor_update"] += time.perf_counter() - t0
            
            t0 = time.perf_counter()
            result = self._select_actions(obs_dict, training=True, return_infos=True)
            if profile_timing:
                timing["select_actions"] += time.perf_counter() - t0
            actions, action_infos = result  # type: ignore
            
            t0 = time.perf_counter()
            for agent_id in self.agent_ids:
                if agent_id in obs_dict and agent_id in actions:
                    action = actions[agent_id]
                    info = action_infos.get(agent_id, {}) if isinstance(action_infos, dict) else {}

                    if (not self.disable_corridor_agent) and isinstance(info, dict) and "lambda" in info:
                        try:
                            v_loc = info.get("value_loc", None)
                            v_glob = info.get("value_glob", None)
                            v_loc_f = float(v_loc.item()) if hasattr(v_loc, "item") else None
                            v_glob_f = float(v_glob.item()) if hasattr(v_glob, "item") else None
                            delta_f = (v_glob_f - v_loc_f) if (v_loc_f is not None and v_glob_f is not None) else None

                            agent_obj = self.intersection_agents.get(agent_id)
                            mask_cnt = 0
                            pref_cnt = 0
                            pref_mean = 0.0
                            if agent_obj is not None:
                                try:
                                    mask_cnt = int(sum(1 for v in getattr(agent_obj, "lane_mask", {}).values() if v))
                                except Exception:
                                    mask_cnt = 0
                                try:
                                    pref_vals = list(getattr(agent_obj, "preference_values", {}).values())
                                    pref_cnt = len(pref_vals)
                                    pref_mean = float(np.mean(pref_vals)) if pref_vals else 0.0
                                except Exception:
                                    pref_cnt = 0
                                    pref_mean = 0.0

                            self._append_lambda_log(
                                episode_length, agent_id, info["lambda"],
                                v_loc=v_loc_f, v_glob=v_glob_f, delta=delta_f,
                                lane_mask_count=mask_cnt, pref_count=pref_cnt, pref_mean=pref_mean
                            )
                        except Exception:
                            pass
                    try:
                        lam = float(info.get("lambda", 0.0)) if isinstance(info, dict) else 0.0
                        buffers[agent_id]['lambdas'].append(lam)
                    except Exception:
                        buffers[agent_id]['lambdas'].append(0.0)

                    try:
                        agent_obj = self.intersection_agents.get(agent_id)
                        lane_mask = getattr(agent_obj, "lane_mask", {}) if agent_obj is not None else {}
                        in_corridor = any(bool(v) for v in (lane_mask or {}).values()) if lane_mask else False
                        buffers[agent_id]['in_corridor'].append(bool(in_corridor))
                    except Exception:
                        buffers[agent_id]['in_corridor'].append(False)

                    try:
                        agent_obj = self.intersection_agents.get(agent_id)
                        lane_ids = list(getattr(agent_obj, "lane_ids", []) or [])
                        if not lane_ids:
                            lf = buffers[agent_id]['lane_features'][-1] if buffers[agent_id]['lane_features'] else {}
                            lane_ids = list((lf or {}).keys())
                        lane_mask = getattr(agent_obj, "lane_mask", {}) if agent_obj is not None else {}
                        pref = getattr(agent_obj, "preference_values", {}) if agent_obj is not None else {}
                        mask_part = [1.0 if lane_mask.get(lid, False) else 0.0 for lid in lane_ids]
                        pref_part = [float(pref.get(lid, 0.0)) for lid in lane_ids]
                        buffers[agent_id]['corridor_task_vec'].append(mask_part + pref_part)
                    except Exception:
                        buffers[agent_id]['corridor_task_vec'].append([])
                    if not self.disable_corridor_agent:
                        try:
                            lam = 0.0
                            if isinstance(info, dict) and "lambda" in info:
                                lam = float(info.get("lambda", 0.0))
                            self._append_preference_log(episode_length, agent_id, lam)
                        except Exception:
                            pass
                    
                    if "log_prob" in info and "value_loc" in info:
                        buffers[agent_id]['log_probs'].append(info["log_prob"].item())
                        buffers[agent_id]['values_loc'].append(info["value_loc"].item())
                        buffers[agent_id]['values_glob'].append(info.get("value_glob", info["value_loc"]).item())
                    else:
                        try:
                            obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32)
                            lane_features = self._get_lane_features(agent_id)
                            log_prob, value_loc, value_glob = self.intersection_agents[agent_id].evaluate_action(
                                obs, action, lane_features
                            )
                            buffers[agent_id]['log_probs'].append(log_prob.item())
                            buffers[agent_id]['values_loc'].append(value_loc.item())
                            buffers[agent_id]['values_glob'].append(value_glob.item())
                        except Exception as e:
                            print(f"Warning: evaluate_action failed ({agent_id}): {e}")
                            buffers[agent_id]['log_probs'].append(0.0)
                            buffers[agent_id]['values_loc'].append(0.0)
                            buffers[agent_id]['values_glob'].append(0.0)

                    try:
                        agent_obj = self.intersection_agents.get(agent_id)
                        v = float(getattr(agent_obj, "corridor_q_value", 0.0)) if agent_obj is not None else 0.0
                        buffers[agent_id]['corridor_q_observations'].append(v)
                    except Exception:
                        buffers[agent_id]['corridor_q_observations'].append(0.0)
                    
                    buffers[agent_id]['states'].append(obs_dict[agent_id])
                    buffers[agent_id]['actions'].append(action)
                    if self.collect_lane_features:
                        lane_features = self._get_lane_features(agent_id)
                        buffers[agent_id]['lane_features'].append(lane_features)
                    else:
                        buffers[agent_id]['lane_features'].append({})
            
            if profile_timing:
                timing["post_step"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            step_result = self.env.step(actions)
            if profile_timing:
                timing["env_step"] += time.perf_counter() - t0
            if len(step_result) == 4:
                next_obs_dict, rewards, terminations, truncations = step_result
                infos = {}
            else:
                next_obs_dict, rewards, terminations, truncations, infos = step_result
            
            if isinstance(next_obs_dict, tuple):
                next_obs_dict = next_obs_dict[0]

            collect_metrics = self.config.get("collect_traffic_metrics", True)  # Can be disabled via config
            metrics_collect_interval = self.config.get("metrics_collect_interval", 10)  # Collect every N steps
            
            if collect_metrics and (step_count % metrics_collect_interval == 0):
                try:
                    ts_map = getattr(base_env_for_metrics, "traffic_signals", None)
                    if isinstance(ts_map, dict) and ts_map:
                        step_total_queued = 0
                        for ts in ts_map.values():
                            try:
                                step_total_queued += int(ts.get_total_queued())
                            except Exception:
                                pass
                        queue_total_sum += float(step_total_queued)
                        queue_step_count += 1
                except Exception:
                    pass
            
            for agent_id in self.agent_ids:
                if agent_id in rewards:
                    reward = float(rewards[agent_id])
                    if reward_fn == "queue":
                        try:
                            lf = buffers.get(agent_id, {}).get('lane_features', [])
                            lane_feat = lf[-1] if lf else {}
                            if lane_feat:
                                q_vals = []
                                for feat in lane_feat.values():
                                    feat_t = feat if isinstance(feat, torch.Tensor) else torch.tensor(feat, dtype=torch.float32)
                                    if feat_t.numel() > 1:
                                        q_vals.append(float(feat_t[1].item()))
                                if q_vals:
                                    reward = -float(np.sum(q_vals)) * reward_scale
                        except Exception:
                            pass

                    buffers[agent_id]['rewards'].append(reward)
                    episode_rewards[agent_id] += reward
                    reward_glob = reward
                    try:
                        if (not self.disable_corridor_agent) and reward_glob_mode == "shaping" and corridor_global_reward_coef != 0.0:
                            lane_feat = buffers[agent_id]['lane_features'][-1] if buffers[agent_id]['lane_features'] else {}
                            bonus = _corridor_global_bonus(agent_id, lane_feat)
                            reward_glob = reward + corridor_global_reward_coef * bonus
                        elif reward_glob_mode == "same_as_loc":
                            reward_glob = reward
                    except Exception:
                        reward_glob = reward
                    buffers[agent_id]['rewards_glob'].append(float(reward_glob))
                    try:
                        acc = reward_window_acc.get(agent_id)
                        if acc is not None:
                            if acc["count"] == 0:
                                acc["start_step"] = int(episode_length)
                            acc["sum_loc"] += float(reward)
                            acc["sum_glob"] += float(reward_glob)
                            acc["count"] += 1
                            if acc["count"] >= reward_log_window:
                                end_step = int(episode_length)
                                self._append_reward_log(
                                    step_start=int(acc["start_step"]),
                                    step_end=end_step,
                                    agent_id=agent_id,
                                    reward_loc_mean=float(acc["sum_loc"]) / float(acc["count"]),
                                    reward_glob_mean=float(acc["sum_glob"]) / float(acc["count"]),
                                    window_size=int(acc["count"]),
                                    max_rows=reward_log_max_rows
                                )
                                acc["sum_loc"] = 0.0
                                acc["sum_glob"] = 0.0
                                acc["count"] = 0
                                acc["start_step"] = end_step + 1
                    except Exception:
                        pass
                
                done = bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                buffers[agent_id]['dones'].append(done)
            
            obs_dict = next_obs_dict
            episode_length += 1
            
            self._current_episode_step = episode_length
            if profile_timing:
                timing["steps"] += 1
            
            if episode_length == 1 and step_count > 0 and step_count % 100 == 0:
                print(f"  Step {step_count}...")
                sys.stdout.flush()
        
        avg_reward = np.mean(list(episode_rewards.values()))

        if reward_log_flush_partial:
            try:
                for agent_id, acc in reward_window_acc.items():
                    if acc.get("count", 0) and acc["count"] > 0:
                        end_step = int(episode_length - 1)
                        self._append_reward_log(
                            step_start=int(acc["start_step"]),
                            step_end=end_step,
                            agent_id=agent_id,
                            reward_loc_mean=float(acc["sum_loc"]) / float(acc["count"]),
                            reward_glob_mean=float(acc["sum_glob"]) / float(acc["count"]),
                            window_size=int(acc["count"]),
                            max_rows=reward_log_max_rows
                        )
            except Exception:
                pass
        if profile_timing:
            timing["total"] = (
                timing["corridor_update"]
                + timing["select_actions"]
                + timing["env_step"]
                + timing["post_step"]
            )

        avg_queue_length = float(queue_total_sum) / float(queue_step_count) if queue_step_count > 0 else 0.0
        avg_speed_mps = 0.0
        avg_speed_norm = 1.0
        
        self._training_corridor_components = False
        
        return {
            'buffers': buffers,
            'episode_rewards': episode_rewards,
            'avg_reward': avg_reward,
            'episode_length': episode_length,
            'avg_queue_length': avg_queue_length,
            'avg_speed_mps': avg_speed_mps,
            'avg_speed_norm': avg_speed_norm,
            'timing': timing if profile_timing else None
        }
    
    def _update_agents(self, rollout_data: Dict[str, Any]):
        """Update all agents."""
        buffers = rollout_data.get('buffers', {})
        reward_glob_mode = str(self.config.get("reward_glob_mode", "shaping")).lower()
        
        for agent_id in self.agent_ids:
            if agent_id not in buffers or len(buffers[agent_id]['states']) == 0:
                continue
            
            buf = buffers[agent_id]
            trainer = self.mgda_trainers[agent_id]
            
            rewards_loc = buf['rewards']
            rewards_glob = buf.get('rewards_glob', buf['rewards']) if not self.disable_corridor_agent else buf['rewards']

            original_disable = self.intersection_agents[agent_id].disable_corridor_agent
            try:
                if reward_glob_mode == "same_as_loc":
                    self.intersection_agents[agent_id].disable_corridor_agent = True
                elif (not original_disable) and (not self.disable_corridor_agent):
                    has_corridor_signal = any(abs(float(r)) > 1e-12 for r in rewards_glob) if rewards_glob else False
                    if not has_corridor_signal:
                        self.intersection_agents[agent_id].disable_corridor_agent = True
            except Exception:
                pass
            
            rollout_data_for_trainer = {
                'states': buf['states'],
                'actions': buf['actions'],
                'rewards_loc': rewards_loc,
                'rewards_glob': rewards_glob,
                'values_loc': buf['values_loc'],
                'values_glob': buf['values_glob'] if not self.disable_corridor_agent else buf['values_loc'],
                'dones': buf['dones'],
                'old_log_probs': buf['log_probs'],
                'lambdas': buf.get('lambdas', [0.0] * len(buf.get('states', []))),
                'in_corridor': buf.get('in_corridor', [False] * len(buf.get('states', []))),
                'lane_features': buf['lane_features'],
                'corridor_task_vec': buf.get('corridor_task_vec', [[]] * len(buf.get('states', []))),
                'corridor_q_observations': buf.get('corridor_q_observations', [0.0] * len(buf.get('states', []))),
            }
            
            try:
                trainer.update(rollout_data_for_trainer)
            except Exception as e:
                print(f"Warning: failed to update agent ({agent_id}): {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Restore original mode (avoid affecting the next episode/step behavior)
                try:
                    self.intersection_agents[agent_id].disable_corridor_agent = original_disable
                except Exception:
                    pass
        
        # Update Corridor Agent components (if enabled)
        if (not self.disable_corridor_agent) and (not self.disable_gnn) and (reward_glob_mode != "same_as_loc"):
            self._update_corridor_components(rollout_data)
    
    def _update_corridor_components(self, rollout_data: Dict[str, Any]):
        """
        Update CorridorAgent components (GNN, Seed Module, Grow Module).

        Uses lower-level IntersectionAgent rewards as upper-level reward signals, and aligns each corridor's
        log_prob with the accumulated reward over its corresponding time interval.
        """
        if not hasattr(self, 'corridor_log_probs') or not self.corridor_log_probs:
            return  # No log_probs; skip training
        
        buffers = rollout_data.get('buffers', {})
        if not buffers:
            return

        # Keep reward_glob_metric consistent with reward_fn (upper-level training uses the same metric)
        reward_glob_metric = str(self.config.get("reward_fn", "diff-waiting-time")).lower()
        reward_scale = float(self.config.get("reward_scale", 1.0))

        def _edge_lane_metric(feat: Any, metric: str) -> float:
            """Extract lane-level reward from lane_features (larger = worse, so return negative values)."""
            try:
                if isinstance(feat, torch.Tensor):
                    feat_t = feat
                else:
                    feat_t = torch.tensor(feat, dtype=torch.float32)
                density = float(feat_t[0].item()) if feat_t.numel() > 0 else 0.0
                queue = float(feat_t[1].item()) if feat_t.numel() > 1 else 0.0
                wait = float(feat_t[2].item()) if feat_t.numel() > 2 else 0.0
                speed = float(feat_t[3].item()) if feat_t.numel() > 3 else 0.0
                if metric == "density":
                    val = -float(density)
                elif metric == "queue_density":
                    val = -float(density + queue)
                elif metric == "diff-waiting-time":
                    val = float(wait)
                elif metric == "pressure":
                    val = -float(density)
                elif metric == "average-speed":
                    val = float(speed)
                else:  # "queue"
                    val = -float(queue) / 50.0
                return float(val) * reward_scale
            except Exception:
                return 0.0
        
        # Get step-wise rewards for all agents (used to compute accumulated rewards for aligned intervals)
        # Assumes all agents have rewards of the same length
        first_agent_id = list(buffers.keys())[0]
        num_steps = len(buffers[first_agent_id]['rewards'])
        
        # Compute per-step average reward across all agents
        step_rewards = []
        for step_idx in range(num_steps):
            step_reward_sum = 0.0
            step_reward_count = 0
            for agent_id in self.agent_ids:
                if agent_id in buffers and step_idx < len(buffers[agent_id]['rewards']):
                    step_reward_sum += buffers[agent_id]['rewards'][step_idx]
                    step_reward_count += 1
            if step_reward_count > 0:
                step_rewards.append(step_reward_sum / step_reward_count)
            else:
                step_rewards.append(0.0)
        
        # Compute reward signal for each corridor
        # Note: local rewards are per-step; if we use a discounted sum, the magnitude grows with interval length.
        # Use config to align scale:
        # - corridor_reward_agg = "sum" (default): discounted sum
        # - corridor_reward_agg = "mean": discounted sum / number of steps in interval
        # - corridor_reward_agg = "discounted_mean": discounted sum / sum of discount weights
        # No corridor position decay (weight decay): discount factor is fixed to 1.0.
        gamma = 1.0
        corridor_reward_agg = str(self.config.get("corridor_reward_agg", "sum")).lower()
        total_loss = torch.tensor(0.0, device=self.device)
        num_valid_corridors = 0
        corridor_reward_signals = []  # For scale comparison (debug)
        
        # Get Q and V values (if available)
        has_q_values = hasattr(self, 'corridor_q_values') and self.corridor_q_values
        has_v_values = hasattr(self, 'corridor_v_values') and self.corridor_v_values
        has_pref_tensors = hasattr(self, 'corridor_pref_tensors') and self.corridor_pref_tensors

        # Preference alignment weight for corridor rewards (auxiliary loss)
        pref_weight = float(self.config.get("preference_value_coef", 0.1))

        # Collect valid corridor info first (avoid needing reward range ahead of time to set targets)
        corridor_items = []  # list of dicts: log_prob, reward_signal, q_loss, v_loss, pref_tensors
        
        for corridor_idx, (corridor_log_probs, update_step) in enumerate(
            zip(self.corridor_log_probs, self.corridor_update_steps)
        ):
            if not corridor_log_probs:
                continue
            
            # Get corresponding Q and V values
            corridor_q_values = self.corridor_q_values[corridor_idx] if has_q_values and corridor_idx < len(self.corridor_q_values) else []
            corridor_v_value = self.corridor_v_values[corridor_idx] if has_v_values and corridor_idx < len(self.corridor_v_values) else None
            corridor_pref_list = self.corridor_pref_tensors[corridor_idx] if has_pref_tensors and corridor_idx < len(self.corridor_pref_tensors) else []
            
            # Compute rewards at the edge level for edges selected by this corridor (edge-level, interval-aligned).
            # Alignment: for each edge (corresponding to a direction node), aggregate only that node's lanes'
            # step-wise rewards over [update_step, next_update_step) to obtain the edge's feedback signal.
            # This matches credit assignment better than assigning the same total corridor return to every step.
            next_update_step = (
                self.corridor_update_steps[corridor_idx + 1]
                if corridor_idx + 1 < len(self.corridor_update_steps)
                else num_steps
            )
            try:
                seg_start = int(update_step)
            except Exception:
                seg_start = 0
            seg_start = max(0, min(seg_start, num_steps))
            seg_end = max(seg_start, min(int(next_update_step), num_steps))
            init_step = max(0, seg_start - 1)

            # Pre-build lane_id -> feat maps for each step in the interval (merged across all intersections)
            lane_feat_by_step: Dict[int, Dict[str, Any]] = {}
            for t in range(init_step, seg_end):
                m: Dict[str, Any] = {}
                for aid in self.agent_ids:
                    lf_seq = buffers.get(aid, {}).get('lane_features', [])
                    if not lf_seq:
                        continue
                    idx = min(t, len(lf_seq) - 1)
                    for lane_id, feat in (lf_seq[idx] or {}).items():
                        m[lane_id] = feat
                lane_feat_by_step[t] = m

            def _edge_step_lane_reward(
                feat: Any,
                metric: str,
                lane_id: str,
                last_wait: Dict[str, float]
            ) -> float:
                """Step-wise lane reward (uses lane_features; diff-waiting-time diffs via local last_wait)."""
                try:
                    if isinstance(feat, torch.Tensor):
                        feat_t = feat
                    else:
                        feat_t = torch.tensor(feat, dtype=torch.float32)
                    density = float(feat_t[0].item()) if feat_t.numel() > 0 else 0.0
                    queue = float(feat_t[1].item()) if feat_t.numel() > 1 else 0.0
                    wait = float(feat_t[2].item()) if feat_t.numel() > 2 else 0.0  # Already scaled by /100.0
                    speed = float(feat_t[3].item()) if feat_t.numel() > 3 else 0.0  # 0~1

                    if metric == "diff-waiting-time":
                        last = float(last_wait.get(lane_id, wait))
                        val = last - wait
                        last_wait[lane_id] = wait
                    elif metric == "waiting-time-reduction":
                        last = float(last_wait.get(lane_id, wait))
                        val = max(last - wait, 0.0)
                        last_wait[lane_id] = wait
                    elif metric == "density":
                        val = -density
                    elif metric == "queue_density":
                        val = -(density + queue)
                    elif metric == "pressure":
                        val = -density
                    elif metric == "average-speed":
                        val = speed
                    else:  # "queue"
                        val = -queue
                    return float(val) * reward_scale
                except Exception:
                    return 0.0

            # Compute each edge's reward based on corridor direction_nodes (per edge: interval mean)
            # Note: if a step has no lane data, filling 0.0 can become a "fake good signal" under negative rewards.
            # Therefore we use edge_valid_mask to exclude missing-data steps from upper-level policy/Q/V training.
            edge_rewards: List[float] = []
            edge_valid_mask: List[bool] = []
            try:
                direction_nodes = list(getattr(self.current_corridors[corridor_idx], "direction_nodes", []) or [])
            except Exception:
                direction_nodes = []

            num_edges = min(len(corridor_log_probs), max(0, len(direction_nodes) - 1))
            for step_i in range(num_edges):
                node = direction_nodes[step_i + 1] if (step_i + 1) < len(direction_nodes) else None
                if node is None or self.graph_builder is None:
                    edge_rewards.append(0.0)
                    edge_valid_mask.append(False)
                    continue
                lane_ids = list(self.graph_builder.direction_to_lanes.get(node, []) or [])
                if not lane_ids or seg_end <= seg_start:
                    edge_rewards.append(0.0)
                    edge_valid_mask.append(False)
                    continue

                # Initialize last_wait for diff-waiting-time: use the wait value from the step before seg_start (if available)
                last_wait: Dict[str, float] = {}
                if reward_glob_metric == "diff-waiting-time":
                    init_feats = lane_feat_by_step.get(init_step, {})
                    for lid in lane_ids:
                        feat0 = init_feats.get(lid)
                        if feat0 is None:
                            continue
                        try:
                            feat0_t = feat0 if isinstance(feat0, torch.Tensor) else torch.tensor(feat0, dtype=torch.float32)
                            w0 = float(feat0_t[2].item()) if feat0_t.numel() > 2 else 0.0
                            last_wait[str(lid)] = w0
                        except Exception:
                            continue

                step_vals: List[float] = []
                for t in range(seg_start, seg_end):
                    feats_t = lane_feat_by_step.get(t, {})
                    lane_vals = []
                    for lid in lane_ids:
                        feat_t = feats_t.get(lid)
                        if feat_t is None:
                            continue
                        lane_vals.append(float(_edge_step_lane_reward(feat_t, reward_glob_metric, str(lid), last_wait)))
                    if lane_vals:
                        # Sum across lanes (paper: sum over lanes for each movement reward).
                        step_vals.append(float(np.sum(lane_vals)))

                if step_vals:
                    edge_rewards.append(float(np.mean(step_vals)))
                    edge_valid_mask.append(True)
                else:
                    edge_rewards.append(0.0)
                    edge_valid_mask.append(False)

            valid_count = int(sum(1 for v in edge_valid_mask if v))
            if valid_count <= 0:
                # This corridor update has no usable lane signal in the interval; skip to avoid 0.0 fake-good contamination.
                continue

            # Cumulative reward (for Q/V): discounted sum of edge-level rewards (only valid steps)
            cumulative_reward = 0.0
            for step_i, r in enumerate(edge_rewards):
                if step_i < len(edge_valid_mask) and (not edge_valid_mask[step_i]):
                    continue
                cumulative_reward += (gamma ** step_i) * float(r)

            # Align scale according to aggregation mode (only affects Q/V targets)
            segment_len = max(1, valid_count)
            if corridor_reward_agg == "mean":
                cumulative_reward = cumulative_reward / float(segment_len)
            elif corridor_reward_agg == "discounted_mean":
                if abs(float(gamma) - 1.0) < 1e-8:
                    denom = float(segment_len)
                else:
                    denom = float((1.0 - (gamma ** segment_len)) / (1.0 - gamma))
                if denom > 0:
                    cumulative_reward = cumulative_reward / denom

            reward_signal = torch.tensor(cumulative_reward, dtype=torch.float32, device=self.device)
            corridor_reward_signals.append(float(cumulative_reward))
            
            # Compute total log_prob for this corridor
            # Important: ensure log_probs keep gradients; do not break the computation graph.
            corridor_log_probs_tensors = []
            for log_prob in corridor_log_probs:
                if isinstance(log_prob, torch.Tensor):
                    if log_prob.device != self.device:
                        log_prob = log_prob.to(self.device)
                    if log_prob.requires_grad:
                        corridor_log_probs_tensors.append(log_prob)
                    else:
                        # If log_prob has no gradient, check whether it comes from the computation graph.
                        # If grad_fn is None, it was detached and gradients cannot be recovered.
                        if log_prob.grad_fn is None:
                            print("Warning: log_prob has no grad_fn (may be detached).")
                            # Skip this log_prob to avoid using invalid gradients
                            continue
                        else:
                            # Try to re-enable gradients (this will not restore graph connectivity)
                            corridor_log_probs_tensors.append(log_prob.requires_grad_(True))
                else:
                    print(f"Warning: log_prob is not a tensor (type={type(log_prob)}).")
                    # Non-tensor log_prob cannot participate in gradient computation; skip
                    continue
            
            if not corridor_log_probs_tensors:
                continue
            
            # Sum all log_probs (keep gradients)
            corridor_total_log_prob = sum(corridor_log_probs_tensors)
            
            # Check whether it's zero (use detach to avoid affecting gradient computation)
            if isinstance(corridor_total_log_prob, torch.Tensor):
                if corridor_total_log_prob.detach().item() == 0.0:
                    continue
            else:
                # If it's not a tensor, compare directly
                if float(corridor_total_log_prob) == 0.0:
                    continue
            
            # 1) Policy loss: align per-edge rewards with corresponding log_probs
            # If edge_rewards is shorter, align by minimum length
            policy_loss = torch.tensor(0.0, device=self.device)
            if corridor_log_probs_tensors:
                for i, (log_prob, r) in enumerate(zip(corridor_log_probs_tensors, edge_rewards)):
                    if i < len(edge_valid_mask) and (not edge_valid_mask[i]):
                        continue
                    r_t = torch.tensor(float(r), device=self.device)
                    policy_loss = policy_loss + (-log_prob * r_t)
            
            # 2) Q loss: Q(s, a) should predict cumulative reward
            q_loss = torch.tensor(0.0, device=self.device)
            if corridor_q_values and len(corridor_q_values) > 0:
                # Convert Q values to tensors and ensure correct device
                q_values_tensors = []
                for q_val in corridor_q_values:
                    if isinstance(q_val, torch.Tensor):
                        if q_val.device != self.device:
                            q_val = q_val.to(self.device)
                        q_values_tensors.append(q_val)
                    else:
                        q_values_tensors.append(torch.tensor(float(q_val), device=self.device, requires_grad=False))
                
                if q_values_tensors:
                    # Compute average Q value (or use the last Q value as a target)
                    avg_q = sum(q_values_tensors) / len(q_values_tensors)
                    # Q-value loss: MSE between predicted Q and actual return
                    q_loss = (avg_q - reward_signal) ** 2
            
            # 3) V loss: V(s) should predict cumulative reward
            v_loss = torch.tensor(0.0, device=self.device)
            if corridor_v_value is not None:
                if isinstance(corridor_v_value, torch.Tensor):
                    if corridor_v_value.device != self.device:
                        corridor_v_value = corridor_v_value.to(self.device)
                    # V-value loss: MSE between predicted V and actual return
                    v_loss = (corridor_v_value - reward_signal) ** 2
                else:
                    v_loss = torch.tensor(0.0, device=self.device)

            corridor_items.append({
                "policy_loss": policy_loss,
                "q_loss": q_loss,
                "v_loss": v_loss,
                "reward_signal": reward_signal,
                "pref_tensors": corridor_pref_list,
                # For preference distribution alignment: per-edge reward contributions (aligned with log_prob / pref_tensors)
                "edge_rewards": list(edge_rewards) if isinstance(edge_rewards, (list, tuple)) else [],
            })

        if not corridor_items:
            return  # No valid corridors

        # Align reward -> target preference (0~1)
        # Use running mean/std + sigmoid(z * temp) so small variations still spread targets.
        rewards_t = torch.stack([it["reward_signal"] for it in corridor_items]).detach()
        r_mean = rewards_t.mean()
        # Note: torch.std defaults to unbiased=True; with <=1 sample it can return NaN (DoF issue)
        if rewards_t.numel() <= 1:
            r_std = torch.zeros_like(r_mean)
        else:
            r_std = rewards_t.std(unbiased=False)
        try:
            if not torch.isfinite(r_std).all():
                r_std = torch.zeros_like(r_mean)
        except Exception:
            r_std = torch.zeros_like(r_mean)

        # Track global mean/std with EMA to reduce small-sample noise
        ema_beta = float(self.config.get("pref_target_ema", 0.9))
        if not hasattr(self, "_pref_reward_mean"):
            self._pref_reward_mean = float(r_mean.item())
            _std0 = float(r_std.item()) if hasattr(r_std, "item") else 0.0
            if not np.isfinite(_std0):
                _std0 = 0.0
            self._pref_reward_std = float(max(_std0, 1e-6))
        else:
            self._pref_reward_mean = (
                ema_beta * float(self._pref_reward_mean) + (1.0 - ema_beta) * float(r_mean.item())
            )
            self._pref_reward_std = (
                ema_beta * float(self._pref_reward_std) + (1.0 - ema_beta) * float(max(float(r_std.item()) if np.isfinite(float(r_std.item())) else 0.0, 1e-6))
            )

        denom = max(float(self._pref_reward_std), 1e-6)
        z = (rewards_t - float(self._pref_reward_mean)) / denom
        temp = float(self.config.get("pref_target_temp", 2.0))
        targets = torch.sigmoid(z * temp)

        target_mean = float(targets.mean().item())
        try:
            _rr = float(r_std.item())
            reward_range = float(_rr) if np.isfinite(_rr) else 0.0
        except Exception:
            reward_range = 0.0

        # Total loss: policy + Q + V + (optional) preference alignment loss
        q_weight = float(self.config.get("q_value_coef", 0.5))
        v_weight = float(self.config.get("v_value_coef", 0.5))

        total_loss = torch.tensor(0.0, device=self.device)
        num_valid_corridors = 0
        total_pref_loss = torch.tensor(0.0, device=self.device)
        pref_count = 0

        for i, it in enumerate(corridor_items):
            corridor_loss = it["policy_loss"] + q_weight * it["q_loss"] + v_weight * it["v_loss"]

            # Preference alignment (distribution version):
            # - Preference is constrained to sum to 1 on each corridor, and lower-level allocation is preference * total_reward
            # - So we no longer pull each preference toward the same scalar target
            # - Instead, align the predicted preference distribution (normalized) to the edge-reward contribution distribution (normalized)
            pref_list = it.get("pref_tensors", []) or []
            if pref_weight > 0.0 and pref_list:
                pref_ts = []
                for p in pref_list:
                    if isinstance(p, torch.Tensor):
                        if p.device != self.device:
                            p = p.to(self.device)
                        # Ensure preference tensor keeps gradients
                        if not p.requires_grad:
                            # If the tensor has no gradients, it may have been detached; skip
                            print("Warning: preference tensor has no gradient (may be detached).")
                            continue
                        pref_ts.append(p)
                if pref_ts:
                    # Target distribution: build from edge_rewards contribution magnitudes
                    # Rewards are usually negative (more congestion -> more negative), so use relu(-reward) as weights.
                    edge_rewards = it.get("edge_rewards", []) or []

                    # Length alignment: preference tensors may be longer than available edge_rewards (e.g., missing lane data
                    # or early corridor termination). If lengths differ, compute distribution loss on min_len to avoid
                    # shape mismatches (e.g., 4 vs 3).
                    n_pref = int(len(pref_ts))
                    n_er = int(len(edge_rewards)) if isinstance(edge_rewards, (list, tuple)) else 0
                    n = int(min(n_pref, n_er))
                    if n <= 0:
                        # No edge rewards to align; skip preference loss
                        tgt = None
                    else:
                        pref_stack = torch.stack(pref_ts[:n])
                        pref_pos = torch.clamp(pref_stack, min=0.0)
                        pref_sum = pref_pos.sum()
                        if pref_sum > 1e-8:
                            pred_dist = pref_pos / (pref_sum + 1e-8)
                        else:
                            pred_dist = torch.ones_like(pref_pos) / float(max(1, pref_pos.numel()))

                        tgt = None
                        try:
                            er = edge_rewards[:n]
                            er_t = torch.tensor([float(x) for x in er], dtype=pref_pos.dtype, device=self.device)
                            w = torch.relu(-er_t)
                            w_sum = w.sum()
                            if w_sum > 1e-8:
                                tgt = w / (w_sum + 1e-8)
                        except Exception:
                            tgt = None
                        if tgt is None:
                            tgt = torch.ones_like(pref_pos) / float(max(1, pref_pos.numel()))

                        # MSE on distributions (detach target to avoid backprop into reward computation)
                        pref_loss = ((pred_dist - tgt.detach()) ** 2).mean()
                        corridor_loss = corridor_loss + pref_weight * pref_loss
                        total_pref_loss = total_pref_loss + pref_loss.detach()
                        pref_count += int(pref_pos.numel())

            total_loss = total_loss + corridor_loss
            num_valid_corridors += 1
        
        # Check whether total_loss is zero (use detach to avoid affecting gradient computation)
        if total_loss.detach().item() == 0.0:
            return  # No valid loss
        
        # Average loss (optional)
        avg_loss = total_loss / num_valid_corridors
        
        # Zero all optimizers
        if self.gnn_optimizer is not None:
            self.gnn_optimizer.zero_grad()
        if self.seed_module_optimizer is not None:
            self.seed_module_optimizer.zero_grad()
        if self.grow_module_optimizer is not None:
            self.grow_module_optimizer.zero_grad()
        
        # Backprop once; gradients are routed to each component automatically
        # Note: Seed Module does not directly receive gradients (select_top_k returns indices only),
        # but Grow Module and GNN gradients should propagate correctly.
        avg_loss.backward()
        
        # Check whether gradients exist (debug)
        has_gnn_grad = False
        has_grow_grad = False
        has_seed_grad = False
        
        if self.gnn_optimizer is not None:
            # Check whether GNN has gradients
            for param in self.gnn_encoder.parameters():
                if param.grad is not None:
                    has_gnn_grad = True
                    break
            if has_gnn_grad:
                torch.nn.utils.clip_grad_norm_(self.gnn_encoder.parameters(), self.config.get("ppo_max_grad_norm", 0.5))
                self.gnn_optimizer.step()
            else:
                print("Warning: GNN encoder has no gradients.")
        
        if self.grow_module_optimizer is not None:
            # Check whether Grow Module has gradients (especially preference_head)
            has_pref_head_grad = False
            for name, param in self.corridor_agent.grow_module.named_parameters():
                if param.grad is not None:
                    has_grow_grad = True
                    if 'preference_head' in name:
                        has_pref_head_grad = True
                        # Debug: print preference_head gradient norm
                        update_count = getattr(self, "_corridor_update_count", 0)
                        if update_count > 0 and update_count % 10 == 0:
                            grad_norm = param.grad.norm().item()
                            print(f"  preference_head.{name}: grad_norm={grad_norm:.6f}")
            if has_grow_grad:
                torch.nn.utils.clip_grad_norm_(self.corridor_agent.grow_module.parameters(), self.config.get("ppo_max_grad_norm", 0.5))
                self.grow_module_optimizer.step()
                if not has_pref_head_grad and pref_count > 0:
                    print("Warning: Grow Module has grads, but preference_head has none (preference tensor may be detached).")
            else:
                print("Warning: Grow Module has no gradients.")
        
        # Note: Seed Module currently does not participate in gradient computation (select_top_k is deterministic).
        # To train Seed Module, use a differentiable Top-k selection method (e.g., Gumbel-Softmax).
        # For now, we skip updating Seed Module.
        if self.seed_module_optimizer is not None:
            # Check whether Seed Module has gradients (usually not, since select_top_k is non-differentiable)
            for param in self.corridor_agent.seed_module.parameters():
                if param.grad is not None:
                    has_seed_grad = True
                    break
            if has_seed_grad:
                torch.nn.utils.clip_grad_norm_(self.corridor_agent.seed_module.parameters(), self.config.get("ppo_max_grad_norm", 0.5))
                self.seed_module_optimizer.step()
            # If there are no gradients, that's expected (select_top_k does not participate in gradients).
        
        if hasattr(self, '_corridor_update_count'):
            self._corridor_update_count += 1
        else:
            self._corridor_update_count = 1
        # Intentionally keep stdout quiet (no corridor-agent training metrics printing).
    
    def _update_corridors(self):
        """Update upper-level corridors and broadcast lane masks/preferences."""
        if self.disable_gnn or self.disable_corridor_agent:
            return
        
        reward_fn = str(self.config.get("reward_fn", "diff-waiting-time")).lower()
        reward_scale = float(self.config.get("reward_scale", 1.0))
        pressure_dict: Dict[str, float] = {}

        if not hasattr(self, "_corridor_last_lane_waiting_time"):
            self._corridor_last_lane_waiting_time = {}

        lane_feat_cache = getattr(self, "_lane_features_cache", None)
        if isinstance(lane_feat_cache, dict) and lane_feat_cache:
            for _aid, lf in lane_feat_cache.items():
                if not isinstance(lf, dict) or not lf:
                    continue
                for lane_id, feat in lf.items():
                    try:
                        feat_t = feat if isinstance(feat, torch.Tensor) else torch.tensor(feat, dtype=torch.float32)
                        density = float(feat_t[0].item()) if feat_t.numel() > 0 else 0.0
                        queue = float(feat_t[1].item()) if feat_t.numel() > 1 else 0.0
                        wait = float(feat_t[2].item()) if feat_t.numel() > 2 else 0.0  # Already scaled by /100.0
                        speed = float(feat_t[3].item()) if feat_t.numel() > 3 else 0.0  # 0~1

                        if reward_fn == "diff-waiting-time":
                            last = float(self._corridor_last_lane_waiting_time.get(lane_id, wait))
                            val = last - wait
                            self._corridor_last_lane_waiting_time[lane_id] = wait
                        elif reward_fn == "waiting-time-reduction":
                            last = float(self._corridor_last_lane_waiting_time.get(lane_id, wait))
                            val = max(last - wait, 0.0)
                            self._corridor_last_lane_waiting_time[lane_id] = wait
                        elif reward_fn == "density":
                            val = -density
                        elif reward_fn == "queue_density":
                            val = -(density + queue)
                        elif reward_fn == "pressure":
                            val = -density
                        elif reward_fn == "average-speed":
                            val = speed
                        else:  # "queue" or unknown: use queue proxy
                            val = -queue

                        pressure_dict[str(lane_id)] = float(val) * reward_scale
                    except Exception:
                        continue
        else:
            try:
                base_env = self._get_base_env()
                ts_map = getattr(base_env, "traffic_signals", {}) if base_env is not None else {}
                if isinstance(ts_map, dict):
                    for ts in ts_map.values():
                        lanes = list(getattr(ts, "lanes", []) or [])
                        for lane_id in lanes:
                            try:
                                pressure_dict[str(lane_id)] = float(self.graph_builder.compute_pressure(str(lane_id), ts)) * reward_scale
                            except Exception:
                                continue
            except Exception:
                pass
        
        
        pressure_graph = self.graph_builder.save_temporal_graph(0, pressure_dict)
        
        training_mode = getattr(self, '_training_corridor_components', False)
        
        if training_mode:
            self.gnn_encoder.train()
            self.corridor_agent.seed_module.train()
            self.corridor_agent.grow_module.train()
        else:
            self.gnn_encoder.eval()
            self.corridor_agent.seed_module.eval()
            self.corridor_agent.grow_module.eval()
        
        if hasattr(self, 'device'):
            if pressure_graph.x.device != self.device:
                pressure_graph = pressure_graph.to(self.device)
        
        if training_mode:
            node_embeddings = self.gnn_encoder(pressure_graph)  # Allow gradient computation
            if not node_embeddings.requires_grad:
                print("Warning: node_embeddings does not require grad; GNN may not train.")
                node_embeddings = node_embeddings.requires_grad_(True)
        else:
            with torch.no_grad():
                node_embeddings = self.gnn_encoder(pressure_graph)
        
        if training_mode:
            result = self.corridor_agent.generate_corridors(
                pressure_graph, node_embeddings, 
                return_log_probs=True, 
                return_q_values=True,
                return_v_values=True,
                return_pref_tensors=True
            )
            corridors, log_probs, q_values, v_values, pref_tensors = result
            self.current_corridors = corridors
            
            if not hasattr(self, 'corridor_log_probs'):
                self.corridor_log_probs = []
            if not hasattr(self, 'corridor_q_values'):
                self.corridor_q_values = []
            if not hasattr(self, 'corridor_v_values'):
                self.corridor_v_values = []
            if not hasattr(self, 'corridor_pref_tensors'):
                self.corridor_pref_tensors = []
            if not hasattr(self, 'corridor_update_steps'):
                self.corridor_update_steps = []
            
            current_step = getattr(self, '_current_corridor_update_step', 0)
            for log_prob_list, q_val_list, v_val, pref_list in zip(log_probs, q_values, v_values, pref_tensors):
                self.corridor_log_probs.append(log_prob_list)
                self.corridor_q_values.append(q_val_list)
                self.corridor_v_values.append(v_val)
                self.corridor_pref_tensors.append(pref_list)
                self.corridor_update_steps.append(current_step)
        else:
            self.current_corridors = self.corridor_agent.generate_corridors(
                pressure_graph, node_embeddings, return_log_probs=False
            )

        # Log corridor count and lengths (each update)
        try:
            current_step = getattr(self, '_current_corridor_update_step', 0)
            self._append_corridor_log(current_step)
        except Exception:
            pass
        
        self._update_intersection_agents()
    
    def _update_intersection_agents(self):
        """Update lower-level agents' corridor tasks (batched, synchronous optimization)."""
        q_reduce = str(self.config.get("corridor_q_reduce", "mean")).lower()  # max | mean | sum

        base_env = self._get_base_env()
        for agent_id in self.intersection_agents:
            lane_mask = {}
            preference_values = {}
            corridor_q_value = 0.0
            corridor_q_list: List[float] = []
            ts = base_env.traffic_signals.get(agent_id) if hasattr(base_env, "traffic_signals") else None
            if ts:
                junction_id = ts.id
                for idx, corridor in enumerate(self.current_corridors):
                    if junction_id not in corridor.junctions:
                        continue

                    corridor_lanes: set = set()
                    try:
                        direction_nodes = list(getattr(corridor, "direction_nodes", []) or [])
                    except Exception:
                        direction_nodes = []
                    for node in direction_nodes:
                        try:
                            corridor_lanes.update(self.graph_builder.direction_to_lanes.get(node, []) or [])
                        except Exception:
                            continue

                    ts_lanes = list(getattr(ts, "lanes", []) or [])
                    selected_lanes = [lid for lid in ts_lanes if lid in corridor_lanes]
                    for lane_id in selected_lanes:
                        lane_mask[lane_id] = True
                        pref = float(getattr(corridor, "preference_values", {}).get(lane_id, 0.0))
                        if pref > 0.0:
                            if lane_id not in preference_values:
                                preference_values[lane_id] = pref

                    try:
                        q_vals = self.corridor_q_values[idx] if (idx < len(self.corridor_q_values)) else []
                    except Exception:
                        q_vals = []
                    try:
                        direction_nodes = list(getattr(corridor, "direction_nodes", []) or [])
                    except Exception:
                        direction_nodes = []

                    num_edges = min(len(q_vals), max(0, len(direction_nodes) - 1))
                    for step_i in range(num_edges):
                        node = direction_nodes[step_i + 1] if (step_i + 1) < len(direction_nodes) else None
                        if node is None:
                            continue
                        try:
                            node_junction = self.graph_builder.direction_to_junction.get(node)
                        except Exception:
                            node_junction = None
                        if node_junction != junction_id:
                            continue
                        qv = q_vals[step_i]
                        if isinstance(qv, torch.Tensor):
                            try:
                                qv = float(qv.detach().item())
                            except Exception:
                                qv = float(qv.item()) if hasattr(qv, "item") else 0.0
                        else:
                            try:
                                qv = float(qv)
                            except Exception:
                                qv = 0.0
                        corridor_q_list.append(qv)

                if corridor_q_list:
                    if q_reduce == "sum":
                        corridor_q_value = float(np.sum(corridor_q_list))
                    elif q_reduce == "max":
                        corridor_q_value = float(np.max(corridor_q_list))
                    else:
                        corridor_q_value = float(np.mean(corridor_q_list))
            
            self.intersection_agents[agent_id].set_corridor_task(
                lane_mask, preference_values, corridor_q_value=corridor_q_value
            )
    
    def _get_lane_features(self, agent_id: str) -> Dict[str, torch.Tensor]:
        """Get lane features (use real lane IDs; update every step)."""
        if not getattr(self, "collect_lane_features", True):
            return {}
        if not hasattr(self, '_lane_features_cache'):
            self._lane_features_cache = {}
        base_env = self._get_base_env()
        ts = base_env.traffic_signals.get(agent_id) if hasattr(base_env, "traffic_signals") else None
        if not ts:
            return {}
        
        lane_feature_dim = 10
        agent = self.intersection_agents.get(agent_id)
        if agent is not None:
            lane_feature_dim = int(agent.lane_feature_dim)
        
        lane_features = {}
        densities = []
        queues = []
        waits = []
        speeds = []
        if hasattr(ts, "get_lanes_density"):
            try:
                densities = ts.get_lanes_density()
            except Exception:
                densities = []
        if hasattr(ts, "get_lanes_queue"):
            try:
                queues = ts.get_lanes_queue()
            except Exception:
                queues = []
        if hasattr(ts, "get_accumulated_waiting_time_per_lane"):
            try:
                waits = ts.get_accumulated_waiting_time_per_lane()
            except Exception:
                waits = []
        collect_lane_speed = self.config.get("collect_lane_speed", True)  # Can be disabled via config
        if collect_lane_speed:
            try:
                if hasattr(ts, "sumo"):
                    for lane_id in ts.lanes:
                        try:
                            mean_speed = float(ts.sumo.lane.getLastStepMeanSpeed(lane_id))
                            allowed_speed = float(ts.sumo.lane.getAllowedSpeed(lane_id))
                            if allowed_speed <= 0.0:
                                speeds.append(0.0)
                            else:
                                if mean_speed < 0.0:
                                    mean_speed = allowed_speed
                                speeds.append(float(mean_speed / allowed_speed))
                        except Exception:
                            speeds.append(0.0)
            except Exception:
                speeds = []
        else:
            speeds = [0.0] * len(ts.lanes) if hasattr(ts, "lanes") else []
        
        for i, lane_id in enumerate(ts.lanes):
            features = torch.zeros(lane_feature_dim)
            if i < len(densities):
                features[0] = float(densities[i])
            if i < len(queues) and lane_feature_dim > 1:
                features[1] = float(queues[i])
            # diff-waiting-time alignment: store in the 3rd dimension
            if i < len(waits) and lane_feature_dim > 2:
                features[2] = float(waits[i]) / 100.0
            # average-speed alignment: store in the 4th dimension (0~1)
            if i < len(speeds) and lane_feature_dim > 3:
                features[3] = float(speeds[i])
            lane_features[lane_id] = features
        
        self._lane_features_cache[agent_id] = lane_features
        return lane_features
    
    def _shutdown_executors(self):
        """Shut down the thread pool."""
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    def close(self):
        """Release resources (env / thread pool).

        Important when creating trainers multiple times (e.g., hyperparameter search).
        """
        try:
            self._shutdown_executors()
        except Exception:
            pass
        try:
            if getattr(self, "env", None) is not None:
                self.env.close()
        except Exception:
            pass
        try:
            self.env = None
        except Exception:
            pass

    def _reset_episode_logs(self):
        """Reset per-episode corridor/lambda/preference logs."""
        self._episode_corridor_logs = []
        self._episode_lambda_logs = []
        self._episode_preference_logs = []
        self._episode_reward_logs = []

    def _append_corridor_log(self, update_step: int):
        """Log minimal corridor stats per corridor-update step.

        We intentionally keep this lightweight (scalars only) to avoid large logs:
        - corridor_count
        - corridor_length_sum (sum of lengths across corridors)
        """
        try:
            corridors = list(getattr(self, "current_corridors", []) or [])
        except Exception:
            corridors = []

        corridor_count = int(len(corridors))
        length_sum = 0.0
        if corridors:
            for c in corridors:
                try:
                    dn = list(getattr(c, "direction_nodes", []) or [])
                except Exception:
                    dn = []
                # Length definition: number of direction nodes in the corridor (consistent with existing logs).
                length_sum += float(len(dn))

        self._episode_corridor_logs.append({
            "step": int(update_step),
            "corridor_count": float(corridor_count),
            "corridor_length_sum": float(length_sum),
        })

    def _append_lambda_log(
        self,
        step: int,
        agent_id: str,
        lambda_value: float,
        **extra: Any
    ):
        """Log per-step lambda values (optionally with diagnostics such as v_loc/v_glob/delta)."""
        entry: Dict[str, Any] = {
            "step": int(step),
            "agent_id": str(agent_id),
            "lambda": float(lambda_value)
        }
        for k, v in (extra or {}).items():
            if v is None:
                continue
            try:
                if isinstance(v, (bool, int, float, str)):
                    entry[k] = v
                else:
                    entry[k] = float(v)
            except Exception:
                entry[k] = str(v)
        self._episode_lambda_logs.append(entry)

    def _append_reward_log(
        self,
        step_start: int,
        step_end: int,
        agent_id: str,
        reward_loc_mean: float,
        reward_glob_mean: float,
        window_size: int,
        max_rows: int = 50000
    ):
        """Log window-averaged rewards (local / global)."""
        self._episode_reward_logs.append({
            "step_start": int(step_start),
            "step_end": int(step_end),
            "window_size": int(window_size),
            "agent_id": str(agent_id),
            "reward_loc_mean": float(reward_loc_mean),
            "reward_glob_mean": float(reward_glob_mean),
        })
        # Prevent unbounded table growth (keep the latest max_rows entries)
        if isinstance(max_rows, int) and max_rows > 0 and len(self._episode_reward_logs) > max_rows:
            overflow = len(self._episode_reward_logs) - max_rows
            if overflow > 0:
                self._episode_reward_logs = self._episode_reward_logs[overflow:]

    def _append_preference_log(self, step: int, agent_id: str, lambda_value: float):
        """Log per-step lane-level λ (reuses the preference_table schema)."""
        lane_ids: List[str] = []
        try:
            # Prefer the lanes that would have been logged originally: keys of preference_values
            pref = dict(getattr(self.intersection_agents[agent_id], "preference_values", {}) or {})
            lane_ids = [str(lid) for lid in pref.keys()]
        except Exception:
            lane_ids = []
        self._episode_preference_logs.append({
            "step": int(step),
            "agent_id": str(agent_id),
            "lambda": float(lambda_value),
            "lane_ids": lane_ids,
        })

    def _build_corridor_logging_metrics(self) -> Dict[str, Any]:
        """Aggregate per-episode scalar stats (kept minimal)."""
        metrics = {}
        lambda_values = [entry.get("lambda", 0.0) for entry in self._episode_lambda_logs]

        metrics["lambda_mean"] = float(np.mean(lambda_values)) if lambda_values else 0.0
        metrics["lambda_max"] = float(np.max(lambda_values)) if lambda_values else 0.0

        # reward_loc / reward_glob per agent (mean of within-episode window means)
        if self._episode_reward_logs:
            by_agent: Dict[str, Dict[str, List[float]]] = {}
            for entry in self._episode_reward_logs:
                agent_id = str(entry.get("agent_id", ""))
                if not agent_id:
                    continue
                if agent_id not in by_agent:
                    by_agent[agent_id] = {"loc": [], "glob": []}
                try:
                    by_agent[agent_id]["loc"].append(float(entry.get("reward_loc_mean", 0.0)))
                    by_agent[agent_id]["glob"].append(float(entry.get("reward_glob_mean", 0.0)))
                except Exception:
                    continue
            metrics["reward_loc_mean_by_agent"] = {
                aid: float(np.mean(vals["loc"])) if vals["loc"] else 0.0
                for aid, vals in by_agent.items()
            }
            metrics["reward_glob_mean_by_agent"] = {
                aid: float(np.mean(vals["glob"])) if vals["glob"] else 0.0
                for aid, vals in by_agent.items()
            }

        # Corridor stats per update
        if self._episode_corridor_logs:
            counts = []
            length_sum_total = 0.0
            count_total = 0.0
            for entry in self._episode_corridor_logs:
                try:
                    c = float(entry.get("corridor_count", 0.0))
                except Exception:
                    c = 0.0
                try:
                    ls = float(entry.get("corridor_length_sum", 0.0))
                except Exception:
                    ls = 0.0
                counts.append(c)
                length_sum_total += ls
                count_total += c

            metrics["corridor_count_mean"] = float(np.mean(counts)) if counts else 0.0
            metrics["corridor_length_mean"] = float(length_sum_total / count_total) if count_total > 0 else 0.0

        return metrics

    def _persist_episode_logs(self, episode: int) -> None:
        """Persist corridor/lambda/preference logs as JSONL."""
        try:
            log_dir = self.save_dir / "corridor_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"episode_{episode}.jsonl"
            with open(log_path, "w", encoding="utf-8") as f:
                for entry in self._episode_corridor_logs:
                    f.write(json.dumps({"type": "corridor", "episode": episode, **entry}) + "\n")
                for entry in self._episode_lambda_logs:
                    f.write(json.dumps({"type": "lambda", "episode": episode, **entry}) + "\n")
                for entry in self._episode_preference_logs:
                    f.write(json.dumps({"type": "preference", "episode": episode, **entry}) + "\n")
                for entry in self._episode_reward_logs:
                    f.write(json.dumps({"type": "reward", "episode": episode, **entry}) + "\n")
        except Exception as e:
            print(f"Warning: failed to write episode logs: {e}")
    
    def start_episode_timing(self, episode: int):
        """Start episode timing."""
        self.episode_start_times[episode] = time.time()
    
    def end_episode_timing(self, episode: int) -> float:
        """End episode timing and return elapsed duration."""
        if episode in self.episode_start_times:
            start_time = self.episode_start_times[episode]
            duration = time.time() - start_time
            self.episode_durations.append(duration)
            del self.episode_start_times[episode]
            return duration
        return 0.0
    
    def get_avg_episode_duration(self) -> float:
        """Get the average episode duration."""
        if self.episode_durations:
            return float(np.mean(self.episode_durations))
        return 0.0
    
    def _parse_tripinfo_avg_travel_time(self, episode: int) -> Dict[str, float]:
        """
        Parse tripinfo.xml and compute travel-time statistics.

        Simplified version based on MoLLMLight's implementation.
        """
        file_episode = episode + self.sumo_episode_offset
        candidates = [
            f"{self.save_dir}/tripinfo_ep{file_episode}.xml",
            f"{self.save_dir}/tripinfo.xml"  # Single file containing all episodes
        ]
        
        tripinfo_file = next((p for p in candidates if os.path.exists(p)), None)
        if tripinfo_file is None:
            return {}
        
        # Check if the file is empty or too small
        try:
            if os.path.getsize(tripinfo_file) < 100:
                return {}
        except:
            return {}
        
        # Keep a per-episode tripinfo snapshot
        tripinfo_path = Path(tripinfo_file)
        episode_tripinfo_path = tripinfo_path.with_name(f"{tripinfo_path.stem}_ep{file_episode}.xml")
        try:
            shutil.copyfile(tripinfo_path, episode_tripinfo_path)
        except Exception:
            pass  # Ignore copy errors
        
        # Try parsing XML
        try:
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
            
            if root is None:
                return {}
            
            durations = []
            speeds = []
            tripinfo_count = 0
            
            # Iterate tripinfo elements
            for ti in root.iter('tripinfo'):
                tripinfo_count += 1
                dur = ti.attrib.get('duration')
                dist = ti.attrib.get('routeLength') or ti.attrib.get('distance')
                if dur is not None:
                    try:
                        duration_val = float(dur)
                        if duration_val > 0:
                            durations.append(duration_val)
                            if dist is not None:
                                try:
                                    dist_val = float(dist)
                                    if dist_val > 0:
                                        speeds.append(dist_val / duration_val)
                                except (ValueError, TypeError):
                                    pass
                    except (ValueError, TypeError):
                        continue
            
            if durations and tripinfo_count > 0:
                avg_speed_mps_tripinfo = float(np.mean(speeds)) if speeds else 0.0
                return {
                    'avg_travel_time': float(np.mean(durations)),
                    'min_travel_time': float(np.min(durations)),
                    'max_travel_time': float(np.max(durations)),
                    'tripinfo_count': float(tripinfo_count),
                    'avg_speed_mps_tripinfo': avg_speed_mps_tripinfo,
                }
            else:
                return {
                    'avg_travel_time': 0.0,
                    'min_travel_time': 0.0,
                    'max_travel_time': 0.0,
                    'tripinfo_count': float(tripinfo_count),
                    'avg_speed_mps_tripinfo': 0.0,
                }
        
        except Exception:
            # If XML parsing fails, try incremental parsing
            try:
                return self._parse_malformed_xml(tripinfo_file)
            except Exception:
                # If all parsing methods fail, return empty dict
                return {}
    
    def _parse_malformed_xml(self, tripinfo_file: str) -> Dict[str, float]:
        """Parse malformed XML (line-by-line)."""
        import re
        
        try:
            durations = []
            speeds = []
            tripinfo_count = 0
            
            with open(tripinfo_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '<tripinfo' in line and 'duration=' in line:
                        tripinfo_count += 1
                        duration_match = re.search(r'duration="([^"]*)"', line)
                        dist_match = re.search(r'routeLength="([^"]*)"', line)
                        if duration_match:
                            try:
                                duration_val = float(duration_match.group(1))
                                if duration_val > 0:
                                    durations.append(duration_val)
                                    if dist_match:
                                        try:
                                            dist_val = float(dist_match.group(1))
                                            if dist_val > 0:
                                                speeds.append(dist_val / duration_val)
                                        except (ValueError, TypeError):
                                            pass
                            except (ValueError, TypeError):
                                continue
            
            if durations:
                avg_speed_mps_tripinfo = float(np.mean(speeds)) if speeds else 0.0
                return {
                    'avg_travel_time': float(np.mean(durations)),
                    'min_travel_time': float(np.min(durations)),
                    'max_travel_time': float(np.max(durations)),
                    'tripinfo_count': float(tripinfo_count),
                    'avg_speed_mps_tripinfo': avg_speed_mps_tripinfo,
                }
            else:
                return {
                    'avg_travel_time': 0.0,
                    'min_travel_time': 0.0,
                    'max_travel_time': 0.0,
                    'tripinfo_count': float(tripinfo_count),
                    'avg_speed_mps_tripinfo': 0.0,
                }
        except Exception:
            return {}
    
    def _setup_logging(self):
        """Set up wandb logging."""
        if self.config.get('wandb', False):
            try:
                import wandb
                # Check whether wandb is logged in
                try:
                    api_key = wandb.api.api_key
                    if not api_key or api_key == '':
                        print("Warning: wandb is not logged in.")
                        print("  Run: wandb login")
                        print("  Or set: export WANDB_API_KEY=your_api_key")
                        print("  Continuing without wandb logging.")
                        return None
                except Exception as e:
                    print(f"Warning: cannot verify wandb login status: {e}")
                    print("  Continuing, but wandb logging may fail.")
                
                # Initialize wandb
                run = wandb.init(
                    project=self.config.get('wandb_project', 'corridor-rl'),
                    name=self.config.get('wandb_run_name', 'corridor_training'),
                    config=self.config,
                    reinit=True  # Allow re-initialization
                )
                print(f"wandb initialized: project={self.config.get('wandb_project', 'corridor-rl')}, run={run.name}")
                return wandb
            except Exception as e:
                print(f"Warning: wandb init failed: {e}")
                print("  Continuing without wandb logging.")
                return None
        return None
    
    def _log_metrics(self, episode: int, metrics: Dict[str, Any], prefix: str = ""):
        """
        Log metrics to wandb.
        
        Args:
            episode: Episode index.
            metrics: Metrics dict.
            prefix: Optional prefix (to distinguish training vs evaluation metrics).
        """
        if self.logger:
            try:
                # Filter and log key metrics
                wandb_metrics = {}
                
                # Add prefix (if provided)
                def add_prefix(key: str) -> str:
                    return f"{prefix}{key}" if prefix else key
                
                # Core training metrics
                if 'avg_reward' in metrics:
                    wandb_metrics[add_prefix('avg_reward')] = metrics['avg_reward']
                
                # Episode rewards
                episode_rewards = metrics.get('episode_rewards', {})
                for agent_id, reward in episode_rewards.items():
                    wandb_metrics[add_prefix(f'{agent_id}_reward')] = reward

                # reward_loc_mean / reward_glob_mean per agent (within-episode window means)
                reward_loc_by_agent = metrics.get("reward_loc_mean_by_agent", {})
                if isinstance(reward_loc_by_agent, dict):
                    for agent_id, val in reward_loc_by_agent.items():
                        wandb_metrics[add_prefix(f'{agent_id}_reward_loc_mean')] = float(val)
                reward_glob_by_agent = metrics.get("reward_glob_mean_by_agent", {})
                if isinstance(reward_glob_by_agent, dict):
                    for agent_id, val in reward_glob_by_agent.items():
                        wandb_metrics[add_prefix(f'{agent_id}_reward_glob_mean')] = float(val)
                
                # Travel time metrics
                travel_time_keys = ['avg_travel_time', 'min_travel_time', 'max_travel_time', 'std_travel_time']
                for key in travel_time_keys:
                    if key in metrics:
                        wandb_metrics[add_prefix(key)] = metrics[key]
                
                # Traffic metrics (episode average)
                traffic_keys = [
                    'avg_queue_length',
                    'avg_speed_mps',
                    'avg_speed_norm',
                ]
                for key in traffic_keys:
                    if key in metrics:
                        wandb_metrics[add_prefix(key)] = metrics[key]

                # Throughput (from tripinfo)
                throughput_keys = [
                    'throughput_count',
                    'throughput_per_step',
                    'throughput_per_hour',
                ]
                for key in throughput_keys:
                    if key in metrics:
                        wandb_metrics[add_prefix(key)] = metrics[key]

                lambda_keys = ['lambda_mean', 'lambda_max']
                for key in lambda_keys:
                    if key in metrics:
                        wandb_metrics[add_prefix(key)] = metrics[key]

                # Corridor stats
                corridor_keys = ['corridor_count_mean', 'corridor_length_mean']
                for key in corridor_keys:
                    if key in metrics:
                        wandb_metrics[add_prefix(key)] = metrics[key]
                
                # Evaluation metrics (if any)
                eval_keys = ['std_reward', 'min_reward', 'max_reward']
                for key in eval_keys:
                    if key in metrics:
                        wandb_metrics[add_prefix(key)] = metrics[key]

                # Log to wandb
                if wandb_metrics:
                    self.logger.log(wandb_metrics, step=episode)
            except Exception as e:
                print(f"Warning: failed to log to wandb: {e}")
                import traceback
                if self.config.get('verbose', False):
                    traceback.print_exc()
    
    def _should_save_checkpoint(self, episode: int) -> bool:
        """Return whether a checkpoint should be saved."""
        return self.checkpoint_interval > 0 and episode % self.checkpoint_interval == 0
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save a checkpoint."""
        if not self.checkpoint_save_models:
            return
        
        target_dir = self.checkpoint_save_dir / self.checkpoint_method_name / self.checkpoint_map_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        agent_entries = {}
        for agent_id, agent in self.intersection_agents.items():
            model_path = target_dir / f"{agent_id}_episode_{episode}.pth"
            try:
                agent.save(str(model_path))
                agent_entries[agent_id] = {
                    'agent_id': agent_id,
                    'observation_dim': agent.observation_dim,
                    'action_dim': agent.action_dim,
                    'lane_feature_dim': agent.lane_feature_dim,
                    'model_path': str(model_path),
                }
            except Exception as e:
                print(f"Warning: failed to save agent {agent_id}: {e}")
        
        # If Corridor Agent is enabled, also save its components
        corridor_entries = {}
        if not self.disable_corridor_agent and not self.disable_gnn:
            if self.gnn_encoder is not None:
                gnn_path = target_dir / f"gnn_encoder_episode_{episode}.pth"
                try:
                    torch.save({
                        'state_dict': self.gnn_encoder.state_dict(),
                        'config': {
                            'input_dim': self.gnn_encoder.input_dim,
                            'hidden_dim': self.gnn_encoder.hidden_dim,
                            'output_dim': self.gnn_encoder.output_dim,
                            'num_layers': self.gnn_encoder.num_layers,
                            'architecture': self.gnn_encoder.architecture,
                        }
                    }, str(gnn_path))
                    corridor_entries['gnn_encoder'] = str(gnn_path)
                except Exception as e:
                    print(f"Warning: failed to save GNN encoder: {e}")
            
            if self.corridor_agent is not None:
                seed_path = target_dir / f"seed_module_episode_{episode}.pth"
                grow_path = target_dir / f"grow_module_episode_{episode}.pth"
                try:
                    torch.save({
                        'state_dict': self.corridor_agent.seed_module.state_dict(),
                    }, str(seed_path))
                    corridor_entries['seed_module'] = str(seed_path)
                except Exception as e:
                    print(f"Warning: failed to save Seed Module: {e}")
                
                try:
                    torch.save({
                        'state_dict': self.corridor_agent.grow_module.state_dict(),
                    }, str(grow_path))
                    corridor_entries['grow_module'] = str(grow_path)
                except Exception as e:
                    print(f"Warning: failed to save Grow Module: {e}")
        
        if not agent_entries:
            print(f"Warning: episode {episode}: no agent checkpoints were saved")
            return
        
        def _checkpoint_safe(obj: Any, depth: int = 6) -> Any:
            """
            Convert an arbitrary object into a form that can be safely serialized by torch.save/pickle.

            This mainly avoids checkpoint save failures from objects that contain local functions/closures
            (e.g., os.environ via _createenviron.<locals>.encode).
            """
            if depth <= 0:
                return repr(obj)
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
            # Path / numpy scalar / numpy array / torch tensor
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (np.integer, np.floating)):
                try:
                    return obj.item()
                except Exception:
                    return float(obj)
            if isinstance(obj, np.ndarray):
                try:
                    return obj.tolist()
                except Exception:
                    return repr(obj)
            if isinstance(obj, torch.Tensor):
                try:
                    return obj.detach().cpu().tolist()
                except Exception:
                    return repr(obj)
            # mapping / sequence
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    # Keys must also be serializable; converting to string is usually safest.
                    sk = k if isinstance(k, str) else repr(k)
                    # Callables (functions/closures): convert to string to avoid pickle issues.
                    if callable(v):
                        out[sk] = repr(v)
                    else:
                        out[sk] = _checkpoint_safe(v, depth=depth - 1)
                return out
            if isinstance(obj, (list, tuple, set)):
                return [_checkpoint_safe(v, depth=depth - 1) for v in obj]
            # Last resort: if it is picklable, keep it; otherwise use repr().
            try:
                pickle.dumps(obj)
                return obj
            except Exception:
                return repr(obj)

        checkpoint_payload = {
            'episode': episode,
            'timestamp': time.time(),
            'method': self.checkpoint_method_name,
            'map_name': self.checkpoint_map_name,
            'config': _checkpoint_safe(self.config),
            'agents': agent_entries,
        }
        
        if corridor_entries:
            checkpoint_payload['corridor_components'] = corridor_entries
        
        if self.checkpoint_save_training_state:
            checkpoint_payload['training_state'] = {
                'recent_metrics': _checkpoint_safe(self.training_metrics_history[-10:] if self.training_metrics_history else []),
            }
        
        checkpoint_filename = f"{self.checkpoint_method_name}_{self.checkpoint_map_name}_episode_{episode}.pth"
        checkpoint_path = target_dir / checkpoint_filename
        
        try:
            torch.save(checkpoint_payload, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: failed to save checkpoint: {e}")
    
    def _evaluate_agent(self, episode: int, num_episodes: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate agent performance.
        
        Args:
            episode: Current episode index.
            num_episodes: Number of evaluation episodes (if None, use config eval_episodes).
            
        Returns:
            Evaluation metrics dict.
        """
        eval_episodes = num_episodes or self.config.get("eval_episodes", 5)
        
        eval_total_rewards = []
        eval_lengths = []
        eval_travel_times = []
        eval_throughput_counts = []
        eval_throughput_per_hour = []
        
        # Switch to evaluation mode (avoid training-time stochasticity)
        for agent_id, agent in self.intersection_agents.items():
            agent.backbone.eval()
            agent.local_head.eval()
            agent.global_head.eval()
            agent.value_loc_head.eval()
            agent.value_glob_head.eval()
        
        for eval_ep in range(eval_episodes):
            obs_dict = self.env.reset()
            if isinstance(obs_dict, tuple):
                obs_dict = obs_dict[0]
            
            ep_len = 0
            ep_total = 0.0
            ep_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            
            # Reset corridor (if enabled)
            if not self.disable_corridor_agent and not self.disable_gnn:
                self.current_corridors = []
            
            while self.env.agents:
                # Update corridor (if needed)
                if not self.disable_corridor_agent and not self.disable_gnn:
                    # Simplification: update once at the beginning
                    if ep_len == 0:
                        self._update_corridors()
                
                # Select actions (evaluation mode: deterministic)
                actions = self._select_actions(obs_dict, training=False, return_infos=False)
                
                # Environment step
                step_result = self.env.step(actions)
                if len(step_result) == 4:
                    next_obs_dict, rewards, terminations, truncations = step_result
                    infos = {}
                else:
                    next_obs_dict, rewards, terminations, truncations, infos = step_result
                
                if isinstance(next_obs_dict, tuple):
                    next_obs_dict = next_obs_dict[0]
                
                # Accumulate rewards
                for agent_id in self.agent_ids:
                    if agent_id in rewards:
                        ep_rewards[agent_id] += float(rewards[agent_id])
                        ep_total += float(rewards[agent_id])
                
                obs_dict = next_obs_dict
                ep_len += 1
            
            eval_total_rewards.append(ep_total)
            eval_lengths.append(ep_len)
            
            # Parse travel time (if available)
            travel_time_metrics = self._parse_tripinfo_avg_travel_time(episode)
            if travel_time_metrics and 'avg_travel_time' in travel_time_metrics:
                eval_travel_times.append(travel_time_metrics['avg_travel_time'])
            try:
                trip_count = float(travel_time_metrics.get("tripinfo_count", 0.0)) if travel_time_metrics else 0.0
                if trip_count > 0:
                    eval_throughput_counts.append(trip_count)
                    sim_seconds = float(ep_len) * float(getattr(self, "delta_time", 1.0))
                    if sim_seconds > 0:
                        eval_throughput_per_hour.append(trip_count / (sim_seconds / 3600.0))
            except Exception:
                pass
        
        # Restore training mode
        for agent_id, agent in self.intersection_agents.items():
            agent.backbone.train()
            agent.local_head.train()
            agent.global_head.train()
            agent.value_loc_head.train()
            agent.value_glob_head.train()
        
        eval_metrics = {
            'avg_reward': float(np.mean(eval_total_rewards)) if eval_total_rewards else 0.0,
            'std_reward': float(np.std(eval_total_rewards)) if eval_total_rewards else 0.0,
            'min_reward': float(np.min(eval_total_rewards)) if eval_total_rewards else 0.0,
            'max_reward': float(np.max(eval_total_rewards)) if eval_total_rewards else 0.0,
        }
        
        if eval_travel_times:
            eval_metrics['avg_travel_time'] = float(np.mean(eval_travel_times))
            eval_metrics['std_travel_time'] = float(np.std(eval_travel_times))
            eval_metrics['min_travel_time'] = float(np.min(eval_travel_times))
            eval_metrics['max_travel_time'] = float(np.max(eval_travel_times))

        if eval_throughput_counts:
            eval_metrics["throughput_count"] = float(np.mean(eval_throughput_counts))
        if eval_throughput_per_hour:
            eval_metrics["throughput_per_hour"] = float(np.mean(eval_throughput_per_hour))
        
        return eval_metrics
    
    def __del__(self):
        """Destructor."""
        self._shutdown_executors()

