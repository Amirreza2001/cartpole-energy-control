#!/usr/bin/env bash
# WP5: train all SAC agents, two queues in parallel (one CPU thread each).
cd "$(dirname "$0")/.."
queue() { for cfg in "$@"; do set -- $cfg; python -m experiments.train_rl --rho "$1" --seed "$2" --steps 150000; done; }
queue "0.01 0" "0.1 0" "0.3 0" "0.01 2" > results/rl/train_a.log 2>&1 &
queue "0.001 0" "0.03 0" "0.01 1" > results/rl/train_b.log 2>&1 &
wait
echo ALL_DONE >> results/rl/train_a.log
