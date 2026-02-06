"""
CorridorLight: a two-level cooperative traffic signal control framework based on corridor rewards.

Based on the design in framework.md, it implements:
1. Traffic pressure graph construction and GNN encoding
2. Corridor Agent (upper-level corridor agent)
3. Intersection Agent (lower-level intersection signal agent, using a mixed policy)
4. Cooperative training mechanism (MGDA)
"""

from .config import (
    Config, MapConfig, AgentConfig, get_configs, get_configs_by_name, apply_runtime_logging_defaults
)
from .models.pressure_graph import PressureGraphBuilder
from .models.gnn_encoder import GNNEncoder
from .models.seed_module import SeedModule
from .models.grow_module import GrowModule
from .agents.corridor_agent import CorridorAgent, Corridor
from .agents.intersection_agent import IntersectionAgent
from .trainers.mgda_trainer import MGDATrainer
from .trainers.corridor_trainer import CorridorRLTrainer

try:
    from .utils import (
        setup_logging, MetricsTracker, compute_gae, normalize_advantages,
        set_seed, count_parameters, save_checkpoint, load_checkpoint,
        create_summary_table, format_time, get_device_info, EarlyStopping
    )
except ImportError:
    # If the utils module is not available, provide empty stubs.
    setup_logging = None
    MetricsTracker = None
    compute_gae = None
    normalize_advantages = None
    set_seed = None
    count_parameters = None
    save_checkpoint = None
    load_checkpoint = None
    create_summary_table = None
    format_time = None
    get_device_info = None
    EarlyStopping = None

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "Config",
    "MapConfig",
    "AgentConfig",
    "get_configs",
    "get_configs_by_name",
    "apply_runtime_logging_defaults",
    
    # Models
    "PressureGraphBuilder",
    "GNNEncoder",
    "SeedModule",
    "GrowModule",
    
    # Agents
    "CorridorAgent",
    "Corridor",
    "IntersectionAgent",
    
    # Trainers
    "MGDATrainer",
    "CorridorRLTrainer",
    
    # Utilities (if available)
    "setup_logging",
    "MetricsTracker",
    "compute_gae",
    "normalize_advantages",
    "set_seed",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "create_summary_table",
    "format_time",
    "get_device_info",
    "EarlyStopping",
]

