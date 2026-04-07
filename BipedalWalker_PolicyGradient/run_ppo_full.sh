#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="$ROOT_DIR/BipedalWalker_PolicyGradient"
VENV_PY="$ROOT_DIR/venv/bin/python"
CONFIG_PATH="$PROJECT_DIR/config_ppo_full.yaml"
LOG_PATH="$PROJECT_DIR/ppo_full_train.log"
MPL_CACHE="$ROOT_DIR/.mplcache"

mkdir -p "$MPL_CACHE"
mkdir -p "$PROJECT_DIR/ppo_results"

# Clean previous PPO outputs so this full run produces a fresh artifact set.
rm -f "$PROJECT_DIR/ppo_results"/training_history.csv
rm -f "$PROJECT_DIR/ppo_results"/training_history.json
rm -f "$PROJECT_DIR/ppo_results"/rewards.png
rm -f "$PROJECT_DIR/ppo_results"/losses.png
rm -f "$PROJECT_DIR/ppo_results"/entropy.png
rm -f "$PROJECT_DIR/ppo_results"/ppo_checkpoint_ep*.pt
rm -f "$PROJECT_DIR/ppo_results"/ppo_final.pt

if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing virtualenv python: $VENV_PY"
  echo "Create it first: python3 -m venv venv"
  exit 1
fi

echo "Starting full PPO training..."
echo "Config: $CONFIG_PATH"
echo "Log:    $LOG_PATH"
echo "Output: $PROJECT_DIR/ppo_results"

MPLCONFIGDIR="$MPL_CACHE" \
MPLBACKEND="Agg" \
"$VENV_PY" "$PROJECT_DIR/training_script.py" \
  --mode train \
  --config "$CONFIG_PATH" \
  2>&1 | tee "$LOG_PATH"

echo "Training finished."
echo "Expected artifacts:"
echo "  - $PROJECT_DIR/ppo_results/training_history.csv"
echo "  - $PROJECT_DIR/ppo_results/training_history.json"
echo "  - $PROJECT_DIR/ppo_results/rewards.png"
echo "  - $PROJECT_DIR/ppo_results/losses.png"
echo "  - $PROJECT_DIR/ppo_results/entropy.png"
echo "  - $PROJECT_DIR/ppo_results/ppo_final.pt"
