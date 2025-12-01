#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/data_defense/imdb_1k.tsv}"
MLM_PATH="${MLM_PATH:-$SCRIPT_DIR/bert-base-uncased}"
TGT_PATH="${TGT_PATH:-$SCRIPT_DIR/bert-base-uncased-imdb}"
OUTPUT_PATH="${OUTPUT_PATH:-$SCRIPT_DIR/data_defense/imdb_logs.tsv}"

echo "Start running bertattack.py..."

python3 "$SCRIPT_DIR/bertattack.py" \
  --data_path "$DATA_PATH" \
  --mlm_path "$MLM_PATH" \
  --tgt_path "$TGT_PATH" \
  --use_sim_mat 1 \
  --output_dir "$OUTPUT_PATH" \
  --num_label 2 \
  --use_bpe 1 \
  --k 48 \
  --start 0 \
  --end 1000 \
  --threshold_pred_score 0

echo "Execution finished."