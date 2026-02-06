#!/usr/bin/env python3
"""
CorridorLight main entrypoint.
Training script for a two-level cooperative traffic signal control framework based on corridor rewards.

Usage:
    python main_corridor.py --mode corridor --map 4x4-Lucas
    python main_corridor.py --map CorridorLight/configs/maps/cologne8.yaml --agent CorridorLight/configs/agents/corridor.yaml
"""

import os
import sys
import argparse
from pathlib import Path

os.environ["LIBSUMO_AS_TRACI"] = "1"  # faster SUMO execution (libsumo)
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from CorridorLight.config import get_configs, get_configs_by_name, apply_runtime_logging_defaults
from CorridorLight.trainers.corridor_trainer import CorridorRLTrainer


def build_main_parser():
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="CorridorLight - a two-level cooperative traffic signal control framework based on corridor rewards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode corridor --map 4x4-Lucas                    # Name-based: select training mode
  %(prog)s --map CorridorLight/configs/maps/cologne8.yaml \\
           --agent CorridorLight/configs/agents/corridor.yaml  # Split configs: map + agent
        """
    )
    
    parser.add_argument("--mode", type=str, default="corridor", help="Training mode (agent type). Default: corridor")
    parser.add_argument("--map", type=str, help="Map config file (net/route/gui, etc.), YAML path")
    parser.add_argument("--agent", type=str, help="Agent config file (algorithm + hyperparams), YAML path")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (disabled by default)")
    parser.add_argument("--device", type=str, help="Override device (e.g., cpu / cuda / cuda:0)")
    parser.add_argument(
        "--reward",
        type=str,
        help="Override reward_fn (also sync reward_glob_metric)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override traffic_scale (traffic volume scaling; e.g., 0.5/1.0/2.0). With --exp, only run this scale."
    )
    parser.add_argument(
        "--exp",
        action="store_true",
        help="Enable the default experiment: for each scale, run reward={queue,waiting-time}, scale={1.0,1.5,2.0}"
    )
    parser.add_argument(
        "-share",
        "--share",
        action="store_true",
        help="Enable lower-level IntersectionAgent parameter sharing on 4x4 maps (shared policy/critic)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs (default: 1). Each run will close and reopen the SUMO environment."
    )
    
    return parser


def check_environment():
    """Check environment setup."""
    print("Checking environment...")
    
    if "SUMO_HOME" not in os.environ:
        print("Error: SUMO_HOME is not set.")
        print("Set SUMO_HOME to your SUMO installation directory, e.g.:")
        print("  export SUMO_HOME=/path/to/sumo")
        return False
    else:
        print(f"SUMO_HOME: {os.environ['SUMO_HOME']}")
    
    required_packages = ['torch', 'numpy', 'torch_geometric']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package}: OK")
        except ImportError:
            missing_packages.append(package)
            print(f"{package}: MISSING")
    
    if missing_packages:
        print(f"\nInstall missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    try:
        import sumo_rl
        print("sumo_rl: OK")
    except ImportError:
        print("sumo_rl: MISSING")
        print("Install: pip install sumo-rl")
        return False
    
    print("Environment check passed.")
    return True


def main():
    """Main entrypoint."""
    parser = build_main_parser()
    args = parser.parse_args()
    
    if not check_environment():
        sys.exit(1)
    
    try:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Seed: {args.seed}")
        
        def _is_file(p: str) -> bool:
            try:
                return bool(p) and Path(p).exists() and Path(p).is_file()
            except Exception:
                return False
        
        map_is_file = _is_file(args.map) if args.map else False
        agent_is_file = _is_file(args.agent) if args.agent else False
        
        if map_is_file or agent_is_file:
            map_config_file = args.map if map_is_file else None
            map_name = None
            if (not map_is_file) and args.map:
                map_cfg_dir = current_dir / "CorridorLight" / "configs" / "maps"
                map_cfg_path = map_cfg_dir / f"{args.map}.yaml"
                if map_cfg_path.exists():
                    map_config_file = str(map_cfg_path)
                else:
                    map_name = args.map
            cfgs = get_configs(
                map_config_file=map_config_file,
                agent_config_file=args.agent if agent_is_file else None,
                map_name=map_name
            )
            config = cfgs['merged']
            print("Config: split files (map + agent)")
        else:
            map_name = args.map if args.map else "4x4-Lucas"
            cfgs = get_configs_by_name(agent_name=args.mode or "corridor", map_name=map_name)
            config = cfgs['merged']
            print(f"Config: name-based (mode={args.mode or 'corridor'}, map={map_name})")

        config.wandb = bool(args.wandb)
        try:
            config.share_intersection_parameters = bool(args.share)
        except Exception:
            pass
        if args.scale is not None:
            try:
                config.traffic_scale = float(args.scale)
            except Exception:
                pass
        if args.device:
            config.device = args.device

        # Naming base
        if map_is_file and args.map:
            inferred_map_for_name = Path(args.map).stem
        else:
            inferred_map_for_name = args.map if args.map else "4x4-Lucas"
        if agent_is_file and args.agent:
            agent_name_for_naming = Path(args.agent).stem
        else:
            agent_name_for_naming = None if (map_is_file or agent_is_file) else args.mode

        def _apply_run_naming(cfg, exp_suffix: str = ""):
            cfg = apply_runtime_logging_defaults(
                cfg,
                map_name=inferred_map_for_name,
                agent_name=agent_name_for_naming
            )
            if exp_suffix:
                cfg.wandb_run_name = f"{cfg.wandb_run_name}_{exp_suffix}"
                cfg.save_dir = str(Path(cfg.save_dir) / exp_suffix)
                try:
                    if Path(str(cfg.checkpoint_save_dir)).name == "checkpoints":
                        cfg.checkpoint_save_dir = str(Path(cfg.save_dir) / "checkpoints")
                except Exception:
                    pass
            if cfg.wandb:
                base_project = str(getattr(cfg, "wandb_project", "corridor-rl"))
                reward_fn = getattr(cfg, "reward_fn", "diff-waiting-time")
                reward_suffix = reward_fn.replace("-", "_")
                cfg.wandb_project = f"{base_project}-{inferred_map_for_name}-{reward_suffix}"
            return cfg

        def _run_training(cfg, exp_suffix: str = ""):
            cfg = _apply_run_naming(cfg, exp_suffix=exp_suffix)

            import logging
            logging.basicConfig(
                level=getattr(cfg, "log_level"),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)
            print(f"Log level: {cfg.log_level}")

            if args.verbose:
                print("\nTraining config:")
                print(f"  agent_type: {cfg.agent_type}")
                print(f"  max_episodes: {cfg.max_episodes}")
                print(f"  net_file: {cfg.net_file}")
                print(f"  save_dir: {cfg.save_dir}")
                if cfg.wandb:
                    print(f"  wandb_project: {cfg.wandb_project}")
                    print(f"  wandb_run_name: {cfg.wandb_run_name}")

            trainer = None
            try:
                trainer = CorridorRLTrainer(cfg.to_dict())
                print("Trainer created.")

                print("Initializing...")
                trainer.initialize()

                print("Training...")
                trainer.train(num_episodes=cfg.max_episodes)

                print("Done.")
            finally:
                # Always release SUMO/libsumo resources (multi-run in one process).
                try:
                    if trainer is not None:
                        trainer.close()
                except Exception:
                    pass
                if bool(getattr(cfg, "wandb", False)):
                    try:
                        import wandb
                        wandb.finish()  # type: ignore[attr-defined]
                    except Exception:
                        pass

        if bool(args.exp):
            scales = [float(args.scale)] if (args.scale is not None) else [1.0, 1.5, 2.0]
            rewards = ["queue", "waiting-time"]
            for scale in scales:
                for reward in rewards:
                    from CorridorLight.config import Config as _Config
                    cfg = _Config.from_dict(config.to_dict())
                    cfg.traffic_scale = float(scale)
                    cfg.reward_fn = str(reward)
                    cfg.reward_glob_metric = str(reward)
                    exp_suffix = f"scale{scale}_rwd-{reward.replace('-', '_')}"
                    print(f"\n🧪 EXP: {exp_suffix}")
                    _run_training(cfg, exp_suffix=exp_suffix)
            return

        if args.reward:
            config.reward_fn = args.reward
            config.reward_glob_metric = args.reward

        num_runs = max(1, int(args.runs))
        if num_runs > 1:
            print(f"\nRepeating runs: {num_runs}")
            print("Each run will close and reopen the SUMO environment.\n")
        
        for run_idx in range(num_runs):
            if num_runs > 1:
                print(f"\n{'='*80}")
                print(f"Run {run_idx + 1}/{num_runs}")
                print(f"{'='*80}\n")
            
            from CorridorLight.config import Config as _Config
            run_config = _Config.from_dict(config.to_dict())
            
            run_suffix = f"run{run_idx + 1}" if num_runs > 1 else ""
            _run_training(run_config, exp_suffix=run_suffix)
            
            if num_runs > 1 and run_idx < num_runs - 1:
                print("\nCleaning up SUMO resources for next run...")
                try:
                    # libsumo uses a global connection; close explicitly.
                    if os.environ.get("LIBSUMO_AS_TRACI") == "1":
                        try:
                            import sumo_rl.environment.env as sumo_env
                            if sumo_env.LIBSUMO:
                                try:
                                    import libsumo
                                    libsumo.close()
                                except Exception:
                                    try:
                                        import traci
                                        traci.close()
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    else:
                        try:
                            import traci
                            traci.close()
                        except Exception:
                            pass
                    
                    import gc
                    gc.collect()

                    import time
                    time.sleep(0.5)
                    
                    print("Cleanup complete.\n")
                except Exception as e:
                    print(f"Warning: cleanup encountered an issue: {e}\n")
        
        if num_runs > 1:
            print(f"\n{'='*80}")
            print(f"All {num_runs} runs completed.")
            print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

