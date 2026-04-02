"""Shared utilities: config, seeds, metrics, plotting, and device selection."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file into a nested dictionary.

    Args:
        path: Filesystem path to ``config.yaml`` (or any YAML file).

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Integer seed shared across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config_device: str) -> torch.device:
    """
    Resolve the torch device from a config string.

    Args:
        config_device: ``"auto"`` selects CUDA when available, else CPU;
            ``"cuda"`` or ``"cpu"`` force that device.

    Returns:
        A ``torch.device`` instance.
    """
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create a directory if it does not exist and return its resolved path.

    Args:
        path: Directory path to create.

    Returns:
        Resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def moving_average(values: List[float], window: int) -> np.ndarray:
    """
    Compute a trailing moving average with a fixed window size.

    For indices ``i < window - 1``, the average uses all available values
    up to ``i`` (shorter effective window).

    Args:
        values: Sequence of scalar metrics (e.g., episode rewards).
        window: Window length; must be at least 1.

    Returns:
        NumPy array of moving averages with the same length as ``values``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(arr[start : i + 1]))
    return out


def save_metrics_csv(rows: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """
    Save a list of metric dictionaries to a CSV file.

    Keys are taken from the union of all row keys; missing values are empty.

    Args:
        rows: One dict per training episode (or step), with stringable values.
        path: Output ``.csv`` path.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def save_metrics_json(data: Any, path: Union[str, Path]) -> None:
    """
    Save an object as formatted JSON (for training history mirroring).

    Args:
        data: Any JSON-serializable structure.
        path: Output ``.json`` path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_timestamp(prefix: str = "run") -> str:
    """
    Build a UTC timestamp string suitable for checkpoint filenames.

    Args:
        prefix: Optional label prepended to the time token.

    Returns:
        String like ``prefix_20260115_143022``.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def checkpoint_path(
    out_dir: Union[str, Path],
    tag: str,
    ext: str = ".pt",
) -> Path:
    """
    Construct a checkpoint file path inside an output directory.

    Args:
        out_dir: Results directory (e.g., ``reinforce_results``).
        tag: Filename stem (e.g., ``checkpoint_ep100``).
        ext: File extension including dot.

    Returns:
        Full path for the checkpoint file.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}{ext}"


def plot_training_rewards(
    episode_rewards: List[float],
    moving_avg_rewards: List[float],
    out_path: Union[str, Path],
    title: str = "Training rewards",
) -> None:
    """
    Save a matplotlib figure of raw and moving-average episode rewards.

    Args:
        episode_rewards: Per-episode total reward.
        moving_avg_rewards: Moving average aligned with episodes (same length).
        out_path: Path to save the figure (e.g., ``.png``).
        title: Figure title.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(episode_rewards) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards, alpha=0.35, label="Episode reward")
    plt.plot(episodes, moving_avg_rewards, label="Moving average reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_losses(
    policy_losses: List[float],
    value_losses: List[float],
    out_path: Union[str, Path],
    title: str = "Training losses",
) -> None:
    """
    Save a matplotlib figure of policy and value losses over training.

    Args:
        policy_losses: Scalar policy loss per episode (or NaN if skipped).
        value_losses: Scalar value loss per episode.
        out_path: Path to save the figure.
        title: Figure title.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(policy_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, policy_losses, label="Policy loss", alpha=0.8)
    plt.plot(episodes, value_losses, label="Value loss", alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def project_root_from_config(config_path: Union[str, Path]) -> Path:
    """
    Return the directory containing ``config.yaml`` (project script root).

    Args:
        config_path: Path to the loaded config file.

    Returns:
        Parent directory of ``config_path``.
    """
    return Path(config_path).resolve().parent
