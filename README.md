# CorridorLight

Official code release for:

**“CorridorLight: Cooperation as Task Negotiation with Causal Gating for Traffic Signal Control”**

This repository implements a two-level traffic signal control framework:
- **Corridor Agent (upper layer)**: plans a small set of junction-disjoint corridor tasks (seed-and-grow).
- **Intersection Agents (lower layer)**: execute signal control with a local policy and a corridor-aware policy mixed by a **causal gating** coefficient \(\lambda\).

## Requirements

- **Linux** (recommended)
- **Python** 3.10+ (tested with Python 3.12)
- **SUMO** + environment variable `SUMO_HOME` set
- Python packages (typical): `torch`, `numpy`, `torch_geometric`, `sumo-rl`, `pyyaml`

## Quick start

1) Set SUMO:

```bash
export SUMO_HOME=/path/to/sumo
```

2) Train with named configs (recommended):

```bash
python3 main_corridor.py --mode corridor --map cologne8
```

3) Train with explicit config paths:

```bash
python3 main_corridor.py \
  --map CorridorLight/configs/maps/cologne8.yaml \
  --agent CorridorLight/configs/agents/corridor.yaml \
  --mode corridor
```

4) Select device and enable W&B:

```bash
python3 main_corridor.py --mode corridor --map arterial4x4 --device cuda:0 --wandb
```

## Configuration

- Map configs: `CorridorLight/configs/maps/*.yaml`
- Agent configs: `CorridorLight/configs/agents/*.yaml`

Common knobs:
- **Corridor planning**: `max_corridors`, `max_corridor_length`, `seed_top_k`, `corridor_delta_time`
- **Causal gating**: `lambda_friction_coeff`, `lambda_max_initial`, `lambda_mode`
- **Cooperative reward**: `reward_glob_mode` (default: `shaping`), `corridor_global_reward_coef`

## Notes

- The entrypoint is `main_corridor.py`.
- Experiments can be repeated via `--runs N` (each run reopens the SUMO environment).

## Credits

This codebase is developed based on **SUMO-RL** ([`https://github.com/LucasAlegre/sumo-rl`](https://github.com/LucasAlegre/sumo-rl)).
Please refer to that repository for **SUMO / SUMO-RL installation instructions** and environment setup (e.g., `SUMO_HOME`, `LIBSUMO_AS_TRACI`).

