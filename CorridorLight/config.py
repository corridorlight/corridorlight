"""
Configuration management for CorridorLight RL training.
Based on MoLLMLight's configuration management structure.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import datetime


@dataclass
class Config:
    """Main configuration class for CorridorLight."""
    
    net_file: str = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
    route_file: str = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"
    sumocfg_file: str = ""
    additional_files: str = ""
    additional_sumo_cmd: str = ""
    gui: bool = False
    num_seconds: int = 8000
    sumo_output_file: str = "summaries/summary.xml"
    min_green: int = 5
    delta_time: int = 5
    no_libsumo: bool = False
    traffic_scale: float = 1.0
    
    agent_type: str = "corridor"
    max_episodes: int = 1000
    eval_interval: int = 100
    save_interval: int = 1000
    device: str = "cpu"
    num_threads: int = 4
    num_rollouts: int = 1
    profile_timing: bool = False
    
    reward_fn: str = "diff-waiting-time"
    reward_scale: float = 1.0
    
    disable_corridor_agent: bool = False
    disable_gnn: bool = False
    skip_corridor_updates: bool = False
    max_corridors: int = 3
    max_corridor_length: int = 10
    seed_top_k: int = 5
    corridor_delta_time: int = 30
    corridor_reward_agg: str = "sum"  # sum | mean | discounted_mean
    
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 64
    gnn_num_layers: int = 2
    gnn_architecture: str = "GCN"
    gnn_dropout: float = 0.1
    
    seed_projection_dim: int = 64
    
    grow_hidden_dim: int = 128
    grow_max_neighbors: int = 4
    
    hidden_dim: int = 128
    lambda_friction_coeff: float = 1.0
    lambda_delta_eps: float = 1e-6
    lane_feature_dim: int = 10
    share_intersection_parameters: bool = False

    congestion_force_green: bool = False
    congestion_metric: str = "queue"
    congestion_queue_threshold: float = 0.9
    congestion_density_threshold: float = 0.9
    corridor_global_reward_coef: float = 1.0
    corridor_global_reward_metric: str = "queue"
    # Paper-aligned default: cooperative reward shaping (augment local reward).
    # corridor_decayed is deprecated in this codebase.
    reward_glob_mode: str = "shaping"  # shaping | same_as_loc
    reward_glob_metric: str = "queue"
    reward_glob_decay: float = 0.9
    reward_glob_reduce: str = "max"
    reward_log_window: int = 100
    reward_log_flush_partial: bool = True
    reward_log_max_rows: int = 50000
    
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    ppo_lam: float = 0.95
    mgda_epsilon: float = 1e-8
    
    wandb: bool = False
    wandb_project: str = "corridor-rl"
    wandb_run_name: str = "corridor_training"
    save_dir: str = "models"
    log_level: str = "INFO"
    collect_traffic_metrics: bool = True
    metrics_collect_interval: int = 10
    collect_speed_metrics: bool = False
    
    checkpoint_interval: int = 0
    checkpoint_save_dir: str = "checkpoints"
    checkpoint_save_models: bool = True
    checkpoint_save_training_state: bool = True
    
    eval_episodes: int = 5
    eval_save_results: bool = True
    
    q_value_coef: float = 0.5
    v_value_coef: float = 0.5
    
    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration."""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        
        converted_dict = {}
        for key, value in config_dict.items():
            if key not in valid_params:
                continue
                
            if isinstance(value, str):
                if 'e-' in value.lower() or 'e+' in value.lower():
                    try:
                        converted_dict[key] = float(value)
                    except ValueError:
                        converted_dict[key] = value
                else:
                    converted_dict[key] = value
            else:
                converted_dict[key] = value
        
        return cls(**converted_dict)
    
    def save(self, filepath: "str | Path") -> None:
        """Save configuration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, filepath: "str | Path") -> 'Config':
        """Load configuration from file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)


@dataclass
class MapConfig:
    """Map/environment-related configuration."""
    net_file: str = "sumo_rl/nets/4x4-Lucas/4x4.net.xml"
    route_file: str = "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"
    reward_fn: str = "diff-waiting-time"
    reward_scale: float = 1.0
    sumocfg_file: str = ""
    additional_files: str = ""
    additional_sumo_cmd: str = ""
    gui: bool = False
    num_seconds: int = 8000
    min_green: int = 5
    delta_time: int = 5
    no_libsumo: bool = False
    log_level: str = "INFO"
    traffic_scale: float = 1.0

    @classmethod
    def from_file(cls, path: str) -> 'MapConfig':
        data = yaml.safe_load(Path(path).read_text())
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_data = {k: v for k, v in data.items() if k in valid_params}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentConfig:
    """Agent/training-related configuration."""
    agent_type: str = "corridor"
    device: str = "cpu"
    max_episodes: int = 1000
    eval_interval: int = 100
    save_interval: int = 1000
    num_threads: int = 4
    num_rollouts: int = 1
    save_dir: str = "models"
    profile_timing: bool = False
    
    # Checkpoint parameters
    checkpoint_interval: int = 0  # 0 = do not save checkpoints; positive = save every N episodes
    checkpoint_save_dir: str = "checkpoints"
    checkpoint_save_models: bool = True
    checkpoint_save_training_state: bool = True
    
    # Evaluation parameters
    eval_episodes: int = 5  # number of episodes to run during evaluation
    eval_save_results: bool = True  # whether to save evaluation results
    
    # Corridor RL specific
    reward_fn: str = "diff-waiting-time"
    reward_scale: float = 1.0
    disable_corridor_agent: bool = False  # Disable Corridor Agent entirely (single-objective mode)
    disable_gnn: bool = False  # Disable GNN (keep observation consistent with MoLLMLight)
    skip_corridor_updates: bool = False  # Skip corridor updates (performance optimization)
    max_corridors: int = 3
    max_corridor_length: int = 10
    seed_top_k: int = 5
    corridor_delta_time: int = 20
    corridor_reward_agg: str = "sum"
    
    # GNN
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 64
    gnn_num_layers: int = 2
    gnn_architecture: str = "GCN"
    gnn_dropout: float = 0.1
    seed_projection_dim: int = 64
    
    # Grow Module
    grow_hidden_dim: int = 128
    grow_max_neighbors: int = 4
    
    # Intersection Agent
    hidden_dim: int = 128
    lambda_friction_coeff: float = 1.0
    lane_feature_dim: int = 10
    corridor_global_reward_coef: float = 1.0
    corridor_global_reward_metric: str = "queue"
    reward_glob_mode: str = "shaping"
    reward_glob_metric: str = "queue"
    reward_glob_decay: float = 0.9
    reward_glob_reduce: str = "max"
    reward_log_window: int = 100
    reward_log_flush_partial: bool = True
    reward_log_max_rows: int = 50000
    
    # PPO/MGDA
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    ppo_lam: float = 0.95
    mgda_epsilon: float = 1e-8
    
    # Q-value and V-value training coefficients
    q_value_coef: float = 0.5  # Q-value loss weight
    v_value_coef: float = 0.5  # V-value loss weight

    @classmethod
    def from_file(cls, path: str) -> 'AgentConfig':
        data = yaml.safe_load(Path(path).read_text())
        # Only keep parameters defined in the dataclass; filter out unknown keys
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}
        filtered_data = {k: v for k, v in data.items() if k in valid_params}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_configs(
    map_config_file: Optional[str] = None,
    agent_config_file: Optional[str] = None,
    map_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load split map and agent configurations."""
    default_all = Config.default().to_dict()
    map_defaults = {k: default_all[k] for k in ['net_file','route_file','reward_fn','reward_scale','sumocfg_file','additional_files','additional_sumo_cmd','gui','num_seconds','min_green','delta_time','no_libsumo','log_level','traffic_scale']}
    agent_defaults = {k: v for k, v in default_all.items() if k not in map_defaults}

    # Safely load configs and filter unknown fields
    map_data = yaml.safe_load(Path(map_config_file).read_text()) if map_config_file else map_defaults
    agent_data = yaml.safe_load(Path(agent_config_file).read_text()) if agent_config_file else agent_defaults
    
    # Filter unknown fields
    import inspect
    map_sig = inspect.signature(MapConfig.__init__)
    agent_sig = inspect.signature(AgentConfig.__init__)
    map_valid_params = set(map_sig.parameters.keys()) - {'self'}
    agent_valid_params = set(agent_sig.parameters.keys()) - {'self'}
    
    map_filtered = {k: v for k, v in map_data.items() if k in map_valid_params}
    agent_filtered = {k: v for k, v in agent_data.items() if k in agent_valid_params}
    
    map_cfg = MapConfig(**map_filtered)
    agent_cfg = AgentConfig(**agent_filtered)

    # Override paths via map_name (extended map support; see MoLLMLight)
    if map_name:
        map_files = {
            "4x4-Lucas": ("sumo_rl/nets/4x4-Lucas/4x4.net.xml", "sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml"),
            # In 2way-single-intersection/, the net filename is single-intersection.net.xml (not 2way-single-intersection.net.xml)
            "2way-single-intersection": ("sumo_rl/nets/2way-single-intersection/single-intersection.net.xml", "sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml"),
            "2x2grid": ("sumo_rl/nets/2x2grid/2x2.net.xml", "sumo_rl/nets/2x2grid/2x2.rou.xml"),
            # For 3x3grid, the actual filenames are 3x3Grid2lanes.net.xml / routes14000.rou.xml
            "3x3grid": ("sumo_rl/nets/3x3grid/3x3Grid2lanes.net.xml", "sumo_rl/nets/3x3grid/routes14000.rou.xml"),
            "4x4loop": ("sumo_rl/nets/4x4loop/4x4loop.net.xml", "sumo_rl/nets/4x4loop/4x4loop.rou.xml"),
            # For double, the filenames are network.net.xml / flow.rou.xml
            "double": ("sumo_rl/nets/double/network.net.xml", "sumo_rl/nets/double/flow.rou.xml"),
            # Nguyen/OW directories are case-sensitive on Linux
            "Nguyen": ("sumo_rl/nets/Nguyen/nguyentl.net.xml", "sumo_rl/nets/Nguyen/nguyencontext.rou.xml"),
            "NguyenNoTL": ("sumo_rl/nets/Nguyen/nguyenNoTL.net.xml", "sumo_rl/nets/Nguyen/nguyencontext.rou.xml"),
            "OW": ("sumo_rl/nets/OW/OW.net.xml", "sumo_rl/nets/OW/OW-nowait.rou.xml"),
            "nanshan": ("sumo_rl/nets/nanshan/osm.net.xml", "sumo_rl/nets/nanshan/osm_pt.rou.xml"),
            "simple": ("sumo_rl/nets/simple/simple.net.xml", "sumo_rl/nets/simple/simple.rou.xml"),
            "single-intersection": ("sumo_rl/nets/single-intersection/single-intersection.net.xml", "sumo_rl/nets/single-intersection/single-intersection.rou.xml"),
            # RESCO
            "arterial4x4": ("sumo_rl/nets/RESCO/arterial4x4/arterial4x4.net.xml", "sumo_rl/nets/RESCO/arterial4x4/arterial4x4_1.rou.xml"),
            "resco_arterial4x4": ("sumo_rl/nets/RESCO/arterial4x4/arterial4x4.net.xml", "sumo_rl/nets/RESCO/arterial4x4/arterial4x4_1.rou.xml"),
            "resco_cologne1": ("sumo_rl/nets/RESCO/cologne1/cologne1.net.xml", "sumo_rl/nets/RESCO/cologne1/cologne1.rou.xml"),
            "resco_cologne3": ("sumo_rl/nets/RESCO/cologne3/cologne3.net.xml", "sumo_rl/nets/RESCO/cologne3/cologne3.rou.xml"),
            "resco_cologne8": ("sumo_rl/nets/RESCO/cologne8/cologne8.net.xml", "sumo_rl/nets/RESCO/cologne8/cologne8.rou.xml"),
            "cologne8": ("sumo_rl/nets/RESCO/cologne8/cologne8.net.xml", "sumo_rl/nets/RESCO/cologne8/cologne8.rou.xml"),
            "cologne1": ("sumo_rl/nets/RESCO/cologne1/cologne1.net.xml", "sumo_rl/nets/RESCO/cologne1/cologne1.rou.xml"),
            "cologne3": ("sumo_rl/nets/RESCO/cologne3/cologne3.net.xml", "sumo_rl/nets/RESCO/cologne3/cologne3.rou.xml"),
            "grid4x4": ("sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml", "sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml"),
            "resco_grid4x4": ("sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml", "sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml"),
            "ingolstadt1": ("sumo_rl/nets/RESCO/ingolstadt1/ingolstadt1.net.xml", "sumo_rl/nets/RESCO/ingolstadt1/ingolstadt1.rou.xml"),
            "ingolstadt7": ("sumo_rl/nets/RESCO/ingolstadt7/ingolstadt7.net.xml", "sumo_rl/nets/RESCO/ingolstadt7/ingolstadt7.rou.xml"),
            "ingolstadt21": ("sumo_rl/nets/RESCO/ingolstadt21/ingolstadt21.net.xml", "sumo_rl/nets/RESCO/ingolstadt21/ingolstadt21.rou.xml"),
        }
        if map_name in map_files:
            net_file, route_file = map_files[map_name]
            map_cfg.net_file = net_file
            map_cfg.route_file = route_file

    merged = {**map_cfg.to_dict(), **agent_cfg.to_dict()}
    merged_config = Config.from_dict(merged)
    # Force corridor reward function to match map reward_fn
    if getattr(map_cfg, "reward_fn", ""):
        merged_config.reward_fn = map_cfg.reward_fn
        if map_cfg.reward_fn in {"queue", "density", "queue_density"}:
            merged_config.reward_glob_metric = map_cfg.reward_fn

    return {'map': map_cfg, 'agent': agent_cfg, 'merged': merged_config}


def get_configs_by_name(agent_name: str = "corridor", map_name: str = "4x4-Lucas") -> Dict[str, Any]:
    """Load configs from CorridorLight/configs/{agents,maps}/NAME.yaml."""
    base = Path(__file__).parent / "configs"
    agent_path = base / "agents" / f"{agent_name}.yaml"
    map_path = base / "maps" / f"{map_name}.yaml"

    if map_path.exists():
        map_cfg = MapConfig.from_file(str(map_path))
    else:
        # fallback to default
        map_cfg = MapConfig()

    if agent_path.exists():
        agent_cfg = AgentConfig.from_file(str(agent_path))
    else:
        # fallback to default
        agent_cfg = AgentConfig()

    # Merge configs and create a Config object
    merged = {**map_cfg.to_dict(), **agent_cfg.to_dict()}
    # from_dict automatically filters unknown keys
    merged_config = Config.from_dict(merged)
    # Force corridor reward function to match map reward_fn
    if getattr(map_cfg, "reward_fn", ""):
        merged_config.reward_fn = map_cfg.reward_fn
        if map_cfg.reward_fn in {"queue", "density", "queue_density"}:
            merged_config.reward_glob_metric = map_cfg.reward_fn
    return {'map': map_cfg, 'agent': agent_cfg, 'merged': merged_config}


def apply_runtime_logging_defaults(config: Config, map_name: Optional[str] = None, agent_name: Optional[str] = None) -> Config:
    """Override wandb_run_name and save_dir at runtime"""
    resolved_map = map_name or "4x4-Lucas"
    resolved_agent = agent_name or config.agent_type

    ts_hour = datetime.datetime.now().strftime("%Y%m%d_%H")

    # Build a name that includes key hyperparameters
    # Key hyperparameters: reward_fn, reward_glob_metric, ppo_lr, ppo_gamma, etc.
    hparam_parts = []
    
    # Reward type (most important hyperparameter)
    reward_fn = getattr(config, "reward_fn", "diff-waiting-time")
    hparam_parts.append(f"rwd-{reward_fn}")
    
    # Global reward metric (if different from reward_fn)
    reward_glob_metric = getattr(config, "reward_glob_metric", None)
    if reward_glob_metric and reward_glob_metric != reward_fn:
        hparam_parts.append(f"glob-{reward_glob_metric}")
    
    # PPO learning rate (compact encoding)
    ppo_lr = getattr(config, "ppo_lr", 3e-4)
    if abs(ppo_lr - 3e-4) < 1e-6:
        lr_str = "lr3e4"
    elif abs(ppo_lr - 1e-4) < 1e-6:
        lr_str = "lr1e4"
    elif abs(ppo_lr - 5e-4) < 1e-6:
        lr_str = "lr5e4"
    else:
        # Convert scientific notation to a compact string, e.g., 3e-4 -> lr3e4, 1e-3 -> lr1e3
        lr_sci = f"{ppo_lr:.0e}"
        lr_parts = lr_sci.split("e")
        if len(lr_parts) == 2:
            base = lr_parts[0].replace(".", "").replace("+", "")
            exp = lr_parts[1].replace("+", "").replace("-", "m")
            lr_str = f"lr{base}e{exp}"
        else:
            lr_str = f"lr{ppo_lr:.4f}".replace(".", "p")
    hparam_parts.append(lr_str)
    
    # Corridor reward decay was removed; do not encode it in run names.
    
    # GNN architecture (if GNN is enabled)
    if not getattr(config, "disable_gnn", False):
        gnn_arch = getattr(config, "gnn_architecture", "GAT")
        hparam_parts.append(f"gnn-{gnn_arch}")
    
    # Join hyperparameter parts
    hparam_str = "_".join(hparam_parts) if hparam_parts else "default"
    
    # Build the full run name
    config.wandb_run_name = f"{resolved_map}_{resolved_agent}_{hparam_str}_{ts_hour}"
    config.save_dir = str(Path("results") / resolved_map / resolved_agent / ts_hour)
    # Avoid different runs sharing the checkpoints directory (preserve custom paths)
    try:
        if Path(str(config.checkpoint_save_dir)) == Path("checkpoints"):
            config.checkpoint_save_dir = str(Path(config.save_dir) / "checkpoints")
    except Exception:
        pass

    return config

