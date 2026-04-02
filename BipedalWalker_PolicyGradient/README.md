# BipedalWalker Policy Gradient (AISE 4030 Assignment 2)

Project overview: train a continuous Gaussian policy with **REINFORCE + baseline** on `BipedalWalker-v3` using Gymnasium. Hyperparameters and agent selection are driven by `config.yaml`. Training writes metrics, checkpoints, and plots under `reinforce_results/` for later comparison with PPO (Task 3). `ppo_agent.py` and `rollout_buffer.py` are intentional stubs for that future work.

## Project tree

```text
BipedalWalker_PolicyGradient/
├── config.yaml
├── environment.py
├── policy_network.py
├── value_network.py
├── reinforce_agent.py
├── ppo_agent.py
├── rollout_buffer.py
├── training_script.py
├── utils.py
├── README.md
├── requirements.txt
├── reinforce_results/
└── ppo_results/
```

## File responsibilities

| File | Role |
|------|------|
| `config.yaml` | Agent type, environment name/seed, shared network/training settings, REINFORCE and PPO hyperparameters, results paths, optional eval checkpoint path. |
| `environment.py` | Builds `BipedalWalker-v3` via Gymnasium; optional `render_mode`; `reset_env` helper for seeded resets. |
| `policy_network.py` | Gaussian policy: configurable MLP body, `tanh` mean bounds, learnable log-std, reparameterized sampling, log-prob and entropy helpers. |
| `value_network.py` | Scalar `V(s)` with the same body shape as the policy (separate weights). |
| `reinforce_agent.py` | Episodic REINFORCE with return normalization, baseline subtraction, separate optimizers, gradient clipping, save/load. |
| `ppo_agent.py` | **Stub** — raises `NotImplementedError` until PPO is implemented for Task 3. |
| `rollout_buffer.py` | **Stub** — reserved for PPO rollouts. |
| `training_script.py` | Loads config, dispatches `train` or `eval`, logging, CSV/JSON history, plots, checkpoints. |
| `utils.py` | YAML, seeds, device, dirs, moving averages, metrics I/O, plotting. |

## Environment setup

Use Conda as specified for the course environment.

```bash
conda create -n AISE4030_A2 python=3.10 -y
conda activate AISE4030_A2
cd BipedalWalker_PolicyGradient
pip install -r requirements.txt
```

**Windows Box2D / build issues:** If `pip install gymnasium[box2d]` fails to compile native code, install SWIG via Conda and retry the pip install:

```bash
conda install swig -y
pip install gymnasium[box2d]
```

## Verification

Run this from the `BipedalWalker_PolicyGradient` directory (after activating the conda environment). **Include the console output in your assignment report** as evidence of a working setup.

```bash
python -c "
import gymnasium as gym
import torch
env = gym.make('BipedalWalker-v3')
obs, info = env.reset(seed=42)
print('Environment created successfully!')
print('Observation shape:', obs.shape)
print('Action space:', env.action_space)
print('Action range:', env.action_space.low, 'to', env.action_space.high)
print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')
env.close()
print('Setup is complete!')
"
```

You should see `Observation shape: (24,)`, a `Box(-1.0, 1.0, (4,), float32)` action space with bounds `-1.0` to `1.0` on each motor, and a `Device:` line. The assignment submission asks for observation space, action space, and device in your report (Section 9).

## Train REINFORCE

From `BipedalWalker_PolicyGradient` (Section 3.1.4 of the PDF):

```bash
python training_script.py
```

This is equivalent to `python training_script.py --mode train --config config.yaml` (those are the defaults).

The script reads `agent_type` from `config.yaml`. With `agent_type: "reinforce"`, training runs as implemented.

### Config-driven agent selection

- Set `agent_type: "reinforce"` for this implementation (default in the provided `config.yaml`).
- Setting `agent_type: "ppo"` causes `training_script.py` to exit with a clear `NotImplementedError` guiding you to implement PPO later. Do not expect PPO training to run from this pass.

Shared hyperparameters live under `shared:` (discount `gamma`, MLP `hidden_sizes`, `activation`, `max_grad_norm`, `device`). REINFORCE-specific keys live under `reinforce:` (learning rates, `max_episodes`, `checkpoint_interval`, return normalization flag, etc.).

## Where outputs are saved

With the default paths, **REINFORCE** artifacts go to `reinforce_results/`:

- `training_history.csv` and `training_history.json` — fields: `episode`, `episode_reward`, `moving_avg_reward`, `policy_loss`, `value_loss`, `entropy` (mean policy entropy per episode; Section 5.4), `episode_length`, `total_env_steps`
- `rewards.png` — raw episode rewards and moving average
- `losses.png` — policy and value loss per episode
- `reinforce_checkpoint_ep{N}.pt` — periodic checkpoints (`checkpoint_interval`)
- `reinforce_final.pt` — weights after the last episode

`ppo_results/` is created for parity and future PPO runs; it is empty in this pass.

## Evaluation / deployment

Load a checkpoint and run a deterministic (mean) policy for `reinforce.eval_episodes` episodes:

```bash
python training_script.py --mode eval --config config.yaml --checkpoint reinforce_results/reinforce_final.pt
```

Optional rendering (when the environment and your display support it):

```bash
python training_script.py --mode eval --config config.yaml --checkpoint reinforce_results/reinforce_final.pt --render
```

You can also set `run.checkpoint` and `run.render` in `config.yaml` and omit CLI flags where applicable.

## Notes for Task 3 (comparison)

Training history and plots are structured so a future PPO run can mirror the same metrics and save to `ppo_results/` for side-by-side analysis (learning curves, sample efficiency, etc.).

## AI usage log / transparency

**Keep the full text of the grading or development prompt you used with AI tools** and attach or reference it in your report’s **AI Usage Log** appendix, as required by the assignment’s transparency rules.
