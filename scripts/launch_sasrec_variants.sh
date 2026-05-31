#!/usr/bin/env bash
# Launch SASRec variant probes on idle GPUs 1/2/3 (base d64/maxlen50 runs on GPU0).
# Clean single-variable sweep around the base on two key sequential axes:
#   GPU1: d128 maxlen50  (capacity axis: match emb128 -> stronger base)
#   GPU2: d64  maxlen100 (context-length axis: longer, capture power users)
#   GPU3: d64  maxlen20  (context-length axis: shorter, match median seq=21)
# All uniform-split, validation-only, NO Kaggle submission.
set -u
export HOME=/opt/data/home UV_LINK_MODE=copy
export XDG_CONFIG_HOME=/opt/data/home/.config XDG_DATA_HOME=/opt/data/home/.local/share
cd /opt/data/kaggle/kmu-rec-sys-26-steam || exit 1
mkdir -p logs

run() { # $1=tag $2=device $3..=args
  local tag="$1"; local dev="$2"; shift 2
  nohup uv run --python 3.13 --with "torch==2.10.0" --with numpy --with pandas --with scipy \
    python3 scripts/sasrec_probe.py "$@" --seed 42 --device "$dev" \
    --out-dir "artifacts/sasrec_variants/$tag" \
    > "logs/sasrec_${tag}.log" 2>&1 &
  echo "launched $tag dev=$dev pid=$!"
  sleep 2
}

run "d128_L50"  cuda:1 --emb-dim 128 --n-blocks 2 --n-heads 2 --maxlen 50  --dropout 0.2 --lr 1e-3 --reg 1e-5 --epochs 200 --batch-size 512
run "d64_L100"  cuda:2 --emb-dim 64  --n-blocks 2 --n-heads 2 --maxlen 100 --dropout 0.2 --lr 1e-3 --reg 1e-5 --epochs 200 --batch-size 512
run "d64_L20"   cuda:3 --emb-dim 64  --n-blocks 2 --n-heads 2 --maxlen 20  --dropout 0.2 --lr 1e-3 --reg 1e-5 --epochs 200 --batch-size 512
