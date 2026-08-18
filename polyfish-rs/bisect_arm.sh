#!/bin/bash
set -e

# Trainer-corrosion bisect (see notes.md, "beta-gate result" section).
# Repeatedly calls train.py on ONLY the static teachers/ set (no fresh or
# archived self-play data) and evals the resulting checkpoint after each
# call, to isolate whether train.py itself corrodes behavior independent of
# self-play/search. Data never changes across iters, so any decay is on the
# trainer.
#
# Arm is selected by which env vars you set before invoking:
#   Arm A (baseline):            ./bisect_arm.sh
#   Arm B (LR/optimizer restart): ARM_LABEL=B TRAIN_LR=0.00001 ./bisect_arm.sh
#   Arm C (value-trunk interference): ARM_LABEL=C VALUE_LOSS_WEIGHT=0 ./bisect_arm.sh
#   Arm D (value-trunk detach):   ARM_LABEL=D DETACH_VALUE_TRUNK=1 ./bisect_arm.sh
#
# Arm A leaves DETACH_VALUE_TRUNK at train.py's default 0, which is NOT what
# run_training_loop.sh exports (=1); set it explicitly to reproduce production.
#
# Output: bisect_<ARM_LABEL>.csv, one row per iteration.

ARM_LABEL="${ARM_LABEL:-A}"
ITERS="${ARM_ITERS:-8}"
EVAL_GAMES="${EVAL_GAMES:-32}"
OUT_CSV="bisect_${ARM_LABEL}.csv"
EVAL_LOG_DIR="bisect_eval_games_${ARM_LABEL}"

export REPLAY_BUFFER_FILES=0
export RUST_BACKTRACE=1
export PYTHONUNBUFFERED=1

# self_play is linked against libtorch (tch-eval/metal-eval features) with no
# rpath baked in — without this, dyld aborts on load ("no LC_RPATH's found").
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="$(.venv/bin/python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

mkdir -p "$EVAL_LOG_DIR"
# Guard: a stray fresh games file here would get trained on as if it were
# new self-play data, breaking the "static data" premise of this arm.
rm -f games_*.safetensors

if [ ! -f "$OUT_CSV" ]; then
    echo "bisect_iter,loss,policy_loss,value_loss,avg_score,avg_captures,avg_attacks,villages_t2c_first,villages_t2c_p50,avg_moves" > "$OUT_CSV"
fi

echo "Arm $ARM_LABEL: $ITERS iterations, TRAIN_LR=${TRAIN_LR:-<default 0.002>} VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHT:-<default 1.0>} DETACH_VALUE_TRUNK=${DETACH_VALUE_TRUNK:-0}"

for ((i=1; i<=ITERS; i++))
do
    echo "=== Arm $ARM_LABEL iter $i/$ITERS: train on teachers/ only ==="
    .venv/bin/python3 train.py > "$EVAL_LOG_DIR/train_iter${i}.log" 2>&1

    echo "=== Arm $ARM_LABEL iter $i/$ITERS: eval ($EVAL_GAMES games, iteration=999) ==="
    ./target/release/self_play --num-games "$EVAL_GAMES" --mcts-iters 64 --iteration 999 \
        --tribe1 Imperius --tribe2 Imperius --actors 128 --eval-servers 3 \
        > "$EVAL_LOG_DIR/eval_iter${i}.log" 2>&1

    # Eval-only games carry no ongoing value once metrics are extracted below —
    # each is multi-GB at the 30-turn eval curriculum, so keep the (KB-sized)
    # stdout logs but discard the raw game tensors rather than archiving them.
    rm -f games_*.safetensors

    .venv/bin/python3 -c "
import json
t = json.load(open('.last_train_metrics.json'))
s = json.load(open('.last_self_play_metrics.json'))
row = [$i, t['loss'], t['policy_loss'], t['value_loss'], s['avg_score'], s['avg_captures'],
       s['avg_attacks'], s['villages_t2c_first'], s['villages_t2c_p50'], s['avg_moves']]
with open('$OUT_CSV', 'a') as f:
    f.write(','.join(str(x) for x in row) + '\n')
"
    tail -n 1 "$OUT_CSV"
done

echo "Arm $ARM_LABEL complete: $OUT_CSV"
