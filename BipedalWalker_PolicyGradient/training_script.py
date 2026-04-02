"""Config-driven training and evaluation entry point for policy gradient agents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from environment import make_env, reset_env
from reinforce_agent import ReinforceBaselineAgent
from utils import (
    checkpoint_path,
    ensure_dir,
    load_config,
    moving_average,
    plot_training_losses,
    plot_training_rewards,
    project_root_from_config,
    save_metrics_csv,
    save_metrics_json,
    set_seed,
)


def _resolve_results_dir(config: Dict[str, Any], project_root: Path, agent_type: str) -> Path:
    """
    Map agent type to the configured results directory under the project root.

    Args:
        config: Loaded YAML configuration.
        project_root: Directory containing ``config.yaml``.
        agent_type: ``"reinforce"`` or ``"ppo"``.

    Returns:
        Absolute path to the results directory for that agent.
    """
    paths = config.get("paths", {})
    if agent_type == "reinforce":
        rel = paths.get("reinforce_results", "reinforce_results")
    else:
        rel = paths.get("ppo_results", "ppo_results")
    return (project_root / str(rel)).resolve()


def build_agent(
    config: Dict[str, Any], obs_dim: int, action_dim: int
) -> ReinforceBaselineAgent:
    """
    Instantiate the agent implementation selected by ``agent_type``.

    Args:
        config: Full configuration dict.
        obs_dim: Observation size from the environment.
        action_dim: Action size from the environment.

    Returns:
        A concrete agent instance.

    Raises:
        NotImplementedError: If ``agent_type`` is ``ppo`` or unknown.
    """
    agent_type = str(config.get("agent_type", "reinforce")).lower()
    if agent_type == "reinforce":
        return ReinforceBaselineAgent(config, obs_dim, action_dim)
    if agent_type == "ppo":
        raise NotImplementedError(
            'agent_type "ppo" is not implemented yet. Use "reinforce" in '
            "config.yaml for this repository pass, or implement PPO in "
            "ppo_agent.py and wire it here."
        )
    raise NotImplementedError(f'Unknown agent_type: "{agent_type}".')


def run_training_episode(
    env: Any,
    agent: ReinforceBaselineAgent,
) -> Tuple[float, int, float, float, float]:
    """
    Collect one full episode and apply the agent's episodic update (REINFORCE).

    Args:
        env: Gymnasium environment.
        agent: REINFORCE agent with episodic storage.

    Returns:
        Tuple of ``episode_reward, episode_length, policy_loss, value_loss,
        mean_entropy``.
        Losses are NaN if no update occurred (should not happen with positive length).
    """
    obs, _ = reset_env(env)
    done = False
    episode_reward = 0.0
    steps = 0
    while not done:
        action, _ = agent.select_action(obs, stochastic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        agent.record_reward(float(reward))
        episode_reward += float(reward)
        steps += 1
        done = bool(terminated or truncated)

    pl, vl, ent = agent.end_episode_update()
    return episode_reward, steps, pl, vl, ent


def train(config_path: Path) -> None:
    """
    Run the full REINFORCE training loop with logging, checkpoints, and plots.

    Args:
        config_path: Path to ``config.yaml``.
    """
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    paths_cfg = config.get("paths", {})
    ensure_dir(project_root / str(paths_cfg.get("ppo_results", "ppo_results")))
    out_dir = ensure_dir(_resolve_results_dir(config, project_root, "reinforce"))

    seed = int(config["environment"].get("seed", 0))
    set_seed(seed)

    env = make_env(config)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    agent = build_agent(config, obs_dim, action_dim)

    rf = config["reinforce"]
    max_episodes = int(rf["max_episodes"])
    ckpt_every = int(rf["checkpoint_interval"])
    ma_window = int(rf["moving_average_window"])

    history: List[Dict[str, Any]] = []
    ep_rewards: List[float] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []
    total_env_steps = 0

    print(f"Training REINFORCE on {config['environment']['name']} — device: {agent.device}")
    print(f"Saving outputs to: {out_dir}")

    for ep in range(1, max_episodes + 1):
        ep_reward, ep_len, p_loss, v_loss, mean_ent = run_training_episode(env, agent)
        total_env_steps += ep_len
        ep_rewards.append(ep_reward)
        policy_losses.append(p_loss)
        value_losses.append(v_loss)

        ma = moving_average(ep_rewards, ma_window)
        ma_r = float(ma[-1]) if len(ma) else float("nan")

        row: Dict[str, Any] = {
            "episode": ep,
            "episode_reward": ep_reward,
            "moving_avg_reward": ma_r,
            "policy_loss": p_loss,
            "value_loss": v_loss,
            "entropy": mean_ent,
            "episode_length": ep_len,
            "total_env_steps": total_env_steps,
        }
        history.append(row)

        print(
            f"[Ep {ep:5d}] reward={ep_reward:8.3f}  ma_{ma_window}={ma_r:8.3f}  "
            f"p_loss={p_loss:.6f}  v_loss={v_loss:.6f}  ent={mean_ent:.4f}  "
            f"len={ep_len}  steps={total_env_steps}"
        )

        if ckpt_every > 0 and ep % ckpt_every == 0:
            ckpt = checkpoint_path(out_dir, f"reinforce_checkpoint_ep{ep}")
            agent.save(ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    final_ckpt = checkpoint_path(out_dir, "reinforce_final")
    agent.save(final_ckpt)
    print(f"Saved final weights: {final_ckpt}")

    csv_path = out_dir / "training_history.csv"
    json_path = out_dir / "training_history.json"
    save_metrics_csv(history, csv_path)
    save_metrics_json(history, json_path)
    print(f"Wrote history: {csv_path}, {json_path}")

    ma_full = moving_average(ep_rewards, ma_window).tolist()
    reward_plot = out_dir / "rewards.png"
    loss_plot = out_dir / "losses.png"
    plot_training_rewards(
        ep_rewards, ma_full, reward_plot, title="REINFORCE training rewards"
    )
    plot_training_losses(policy_losses, value_losses, loss_plot, title="REINFORCE training losses")
    print(f"Saved plots: {reward_plot}, {loss_plot}")

    env.close()


def evaluate_mode(config_path: Path, checkpoint: Path, render: bool) -> None:
    """
    Load a saved REINFORCE checkpoint and run deterministic evaluation episodes.

    Args:
        config_path: Path to ``config.yaml``.
        checkpoint: Path to a ``.pt`` checkpoint.
        render: Whether to request environment rendering.
    """
    config = load_config(config_path)

    seed = int(config["environment"].get("seed", 0))
    set_seed(seed)

    render_mode = "human" if render else None
    env = make_env(config, render_mode=render_mode)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    agent = build_agent(config, obs_dim, action_dim)
    agent.load(checkpoint)

    rf = config["reinforce"]
    n_eval = int(rf.get("eval_episodes", 5))

    mean_r, mean_len = agent.evaluate(env, n_eval, seed=seed, render=render)
    print(
        f"Evaluation complete — mean reward={mean_r:.4f}, mean length={mean_len:.2f} "
        f"over {n_eval} episodes (checkpoint: {checkpoint})"
    )
    env.close()


def main() -> None:
    """Parse CLI arguments and dispatch train or evaluation."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate policy gradient agents (REINFORCE baseline)."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="train: learning loop; eval: load checkpoint and run deterministic policy",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (relative to cwd or absolute).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint .pt path for eval mode (overrides config run.checkpoint).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable environment rendering during eval when supported.",
    )
    args = parser.parse_args()

    cfg_file = Path(args.config).resolve()
    if not cfg_file.is_file():
        print(f"Config not found: {cfg_file}", file=sys.stderr)
        sys.exit(1)

    pre_cfg = load_config(cfg_file)
    agent_type = str(pre_cfg.get("agent_type", "reinforce")).lower()
    if agent_type == "ppo":
        raise NotImplementedError(
            'agent_type "ppo" is not implemented in this pass. Set agent_type to '
            '"reinforce" in config.yaml.'
        )

    if args.mode == "train":
        train(cfg_file)
    else:
        ckpt: Path | None
        if args.checkpoint:
            ckpt = Path(args.checkpoint).resolve()
        else:
            run_cfg = pre_cfg.get("run") or {}
            cp = run_cfg.get("checkpoint")
            if cp is None:
                print("eval mode requires --checkpoint or run.checkpoint in config.", file=sys.stderr)
                sys.exit(1)
            ckpt = (project_root_from_config(cfg_file) / str(cp)).resolve()
        if not ckpt.is_file():
            print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
            sys.exit(1)
        render = bool(args.render) or bool((pre_cfg.get("run") or {}).get("render", False))
        evaluate_mode(cfg_file, ckpt, render=render)


if __name__ == "__main__":
    main()
