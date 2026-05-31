#!/usr/bin/env bash
# Launch the 4 missing emb128 full-test seeds on idle GPUs 1/2/3.
# Idempotent: skips a seed whose test.csv already exists.
set -u
export HOME=/opt/data/home
export UV_LINK_MODE=copy
export XDG_CONFIG_HOME=/opt/data/home/.config
export XDG_DATA_HOME=/opt/data/home/.local/share
cd /opt/data/kaggle/kmu-rec-sys-26-steam || exit 1
mkdir -p logs

seeds=(11 99 2025 314)
devs=(1 2 3 1)
for i in "${!seeds[@]}"; do
  s="${seeds[$i]}"
  dev="cuda:${devs[$i]}"
  tst="artifacts/lightgcn_emb128L4r3_fulltest/seed${s}/test.csv"
  if [ -f "$tst" ]; then
    echo "skip seed=$s (test.csv exists)"
    continue
  fi
  nohup uv run --python 3.13 --with "torch==2.10.0" --with numpy --with pandas --with scipy \
    python3 scripts/emb128_ens_expand_worker.py --seed "$s" --device "$dev" \
    > "logs/emb128_expand_seed${s}.log" 2>&1 &
  echo "launched seed=$s dev=$dev pid=$!"
  sleep 2
done
