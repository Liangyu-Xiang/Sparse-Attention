#!/bin/bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-runs/intraframe_attention_similarity_$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-saf3r_eval}"
MAX_FRAMES="${MAX_FRAMES:-16}"
LAYERS="${LAYERS:-0-23}"
HEADS="${HEADS:-0-15}"
PATCH_QUERY_SAMPLE_COUNT="${PATCH_QUERY_SAMPLE_COUNT:-64}"
TOPK="${TOPK:-32}"
SAMPLE_MODE="${SAMPLE_MODE:-uniform}"

GPUS=(4 5)
SCENES=(
  courtyard electro kicker pipes relief delivery_area
  facade office playground relief_2 terrains
)

mkdir -p "$RUN_ROOT"
printf '%s\n' "${SCENES[@]}" > "$RUN_ROOT/all_scenes.txt"

for worker_id in "${!GPUS[@]}"; do
  gpu="${GPUS[$worker_id]}"
  worker_dir="$RUN_ROOT/worker_${worker_id}_gpu${gpu}"
  mkdir -p "$worker_dir"

  scenes=()
  for idx in "${!SCENES[@]}"; do
    if [ $((idx % ${#GPUS[@]})) -eq "$worker_id" ]; then
      scenes+=("${SCENES[$idx]}")
    fi
  done

  worker_script="$worker_dir/run.sh"
  log_file="$worker_dir/log.txt"
  output_json="$worker_dir/results.json"
  output_csv="$worker_dir/results.csv"

  cat > "$worker_script" <<EOF
#!/bin/bash
set -euo pipefail
cd "$(pwd)"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
trap 'status=\$?; echo "exit_status=\$status"; echo "\$status" > "$worker_dir/exit_code.txt"' EXIT

echo "worker_id=$worker_id"
echo "gpu=$gpu"
echo "scenes=${scenes[*]}"
echo "run_dir=$worker_dir"
echo "max_frames=$MAX_FRAMES layers=$LAYERS heads=$HEADS patch_query_sample_count=$PATCH_QUERY_SAMPLE_COUNT topk=$TOPK"

conda run --no-capture-output -n "$CONDA_ENV" python tools/analyze_vggt_intraframe_attention_similarity.py \\
  --scenes ${scenes[*]} \\
  --max-frames "$MAX_FRAMES" \\
  --sample-mode "$SAMPLE_MODE" \\
  --layers "$LAYERS" \\
  --heads "$HEADS" \\
  --patch-query-sample-count "$PATCH_QUERY_SAMPLE_COUNT" \\
  --topk "$TOPK" \\
  --output-json "$output_json" \\
  --output-csv "$output_csv"
EOF

  chmod +x "$worker_script"
  setsid bash "$worker_script" > "$log_file" 2>&1 < /dev/null &
  echo "$!" > "$worker_dir/pid.txt"
  echo "Started worker $worker_id on GPU $gpu: $worker_script"
  echo "  log: $log_file"
done

cat > "$RUN_ROOT/merge.sh" <<EOF
#!/bin/bash
set -euo pipefail
cd "$(pwd)"
conda run --no-capture-output -n "$CONDA_ENV" python tools/analyze_vggt_intraframe_attention_similarity.py \\
  --merge "$RUN_ROOT"/worker_*_gpu*/results.json \\
  --topk "$TOPK" \\
  --output-json "$RUN_ROOT/merged_results.json" \\
  --output-csv "$RUN_ROOT/merged_results.csv"
EOF
chmod +x "$RUN_ROOT/merge.sh"

echo "Run root: $RUN_ROOT"
echo "Merge when workers finish: $RUN_ROOT/merge.sh"
