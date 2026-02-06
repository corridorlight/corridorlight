"""
Utility functions for CorridorLight RL training.
Based on MoLLMLight's utility-function implementation.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import pandas as pd

import torch
import torch.nn as nn


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("CorridorLight")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsTracker:
    """Track and manage training metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.history = []
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.window_size)
            self.metrics[key].append(value)
        
        # Store in history
        self.history.append({
            'step': step,
            'timestamp': time.time(),
            **metrics
        })
    
    def get_mean(self, key: str) -> float:
        """Get mean value of a metric."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return np.mean(self.metrics[key])
    
    def get_std(self, key: str) -> float:
        """Get standard deviation of a metric."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return np.std(self.metrics[key])
    
    def get_latest(self, key: str) -> float:
        """Get latest value of a metric."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return self.metrics[key][-1]
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for key in self.metrics:
            if len(self.metrics[key]) > 0:
                summary[key] = {
                    'mean': np.mean(self.metrics[key]),
                    'std': np.std(self.metrics[key]),
                    'min': np.min(self.metrics[key]),
                    'max': np.max(self.metrics[key]),
                    'latest': self.metrics[key][-1]
                }
        return summary


def compute_gae(rewards: List[float], values: List[float],
                dones: List[bool], gamma: float = 0.99,
                lam: float = 0.95) -> Tuple[Any, Any]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Sequence of rewards
        values: Sequence of value estimates
        dones: Sequence of done flags
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns) with type matching the value input.
    """
    # Determine device/dtype
    if isinstance(values, torch.Tensor):
        device = values.device
        dtype = values.dtype
        values_t = values
    else:
        device = torch.device('cpu')
        dtype = torch.float32
        values_t = torch.as_tensor(values, dtype=dtype, device=device)

    rewards_t = torch.as_tensor(rewards, dtype=dtype, device=device)
    dones_t = torch.as_tensor(dones, dtype=dtype, device=device)

    # Bootstrap value
    values_t = torch.cat([values_t, torch.zeros(1, dtype=dtype, device=device)], dim=0)

    T = rewards_t.shape[0]
    advantages_t = torch.zeros(T, dtype=dtype, device=device)
    gae = torch.tensor(0.0, dtype=dtype, device=device)

    for t in reversed(range(T)):
        mask = 1.0 - dones_t[t]
        delta = rewards_t[t] + gamma * values_t[t + 1] * mask - values_t[t]
        gae = delta + gamma * lam * mask * gae
        advantages_t[t] = gae

    returns_t = advantages_t + values_t[:-1]

    if isinstance(values, torch.Tensor):
        return advantages_t, returns_t

    return advantages_t.tolist(), returns_t.tolist()


def normalize_advantages(advantages: List[float]) -> List[float]:
    """Normalize advantages to have zero mean and unit variance."""
    advantages = np.array(advantages)
    if len(advantages) == 0:
        return advantages.tolist()
    
    mean = np.mean(advantages)
    std = np.std(advantages)
    
    if std == 0:
        return (advantages - mean).tolist()
    
    return ((advantages - mean) / (std + 1e-8)).tolist()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str, **kwargs):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def create_summary_table(metrics: Dict[str, Any]) -> str:
    """Create a formatted summary table from metrics."""
    table = []
    table.append("=" * 50)
    table.append("Training Summary")
    table.append("=" * 50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            table.append(f"{key:30s}: {value:10.4f}")
        else:
            table.append(f"{key:30s}: {value}")
    
    table.append("=" * 50)
    return "\n".join(table)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated(0)
        info['cuda_memory_reserved'] = torch.cuda.memory_reserved(0)
    
    return info


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop early."""
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

