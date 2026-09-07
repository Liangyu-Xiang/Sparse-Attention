#!/bin/bash
set -euo pipefail

CONFIG="${CONFIG:-eval_vggt_eth3d_block3x3_violation_guided.yaml}"
GPUS_CSV="${GPUS:-0,1,2,3}"
RUN_ROOT="${RUN_ROOT:-runs/block3x3_violation_$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-saf3r_eval}"
REFINEMENT_RATIOS="${REFINEMENT_RATIOS:-0.1}"
CANDIDATE_RATIOS="${CANDIDATE_RATIOS:-0.1 0.3 0.5}"
DISPERSION_LAMBDAS="${DISPERSION_LAMBDAS:-0 0.25 0.5 0.75 1.0}"
VIOLATION_ALPHAS="${VIOLATION_ALPHAS:-0 0.25 0.5 0.75 1.0}"
METHODS="${METHODS:-violation}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-8}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "No GPUs configured." >&2
  exit 1
fi

for gpu in "${GPUS[@]}"; do
  if [ "$gpu" -lt 0 ] || [ "$gpu" -gt 5 ]; then
    echo "GPU $gpu is not allowed. Use only GPU IDs 0-5." >&2
    exit 1
  fi
done

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

  scene_override="datasets.eth3d.scenes=[$(IFS=,; echo "${scenes[*]}")]"
  worker_script="$worker_dir/run.sh"
  log_file="$worker_dir/log.txt"

  cat > "$worker_script" <<EOF
#!/bin/bash
set -euo pipefail
cd "$(pwd)"
mkdir -p "$worker_dir/results"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)
trap 'status=\$?; echo "exit_status=\$status"; echo "\$status" > "$worker_dir/exit_code.txt"' EXIT

echo "worker_id=$worker_id"
echo "gpu=$gpu"
echo "scenes=${scenes[*]}"
echo "config=$CONFIG"
echo "run_dir=$worker_dir"

for method in $METHODS; do
  if [ "\$method" = "uniform" ]; then
    "\${PYTHON_CMD[@]}" evaluation/launch.py --config "$CONFIG" \\
      "$scene_override" \\
      "model.patch_module.refinement_strategy=none" \\
      "model.patch_module.candidate_selection=random" \\
      "model.patch_module.pair_selection=random" \\
      "model.patch_module.query_chunk_size=$QUERY_CHUNK_SIZE"
    continue
  fi

  for rr in $REFINEMENT_RATIOS; do
    for rc in $CANDIDATE_RATIOS; do
      if [ "\$method" = "random" ]; then
        "\${PYTHON_CMD[@]}" evaluation/launch.py --config "$CONFIG" \\
          "$scene_override" \\
          "model.patch_module.refinement_strategy=random" \\
          "model.patch_module.candidate_selection=random" \\
          "model.patch_module.pair_selection=random" \\
          "model.patch_module.refinement_ratio=\$rr" \\
          "model.patch_module.candidate_ratio=\$rc" \\
          "model.patch_module.query_chunk_size=$QUERY_CHUNK_SIZE"
        continue
      fi

      for dl in $DISPERSION_LAMBDAS; do
        for va in $VIOLATION_ALPHAS; do
          "\${PYTHON_CMD[@]}" evaluation/launch.py --config "$CONFIG" \\
            "$scene_override" \\
            "model.patch_module.refinement_strategy=violation" \\
            "model.patch_module.candidate_selection=dispersion" \\
            "model.patch_module.pair_selection=optimal" \\
            "model.patch_module.refinement_ratio=\$rr" \\
            "model.patch_module.candidate_ratio=\$rc" \\
            "model.patch_module.dispersion_lambda=\$dl" \\
            "model.patch_module.violation_alpha=\$va" \\
            "model.patch_module.query_chunk_size=$QUERY_CHUNK_SIZE"
        done
      done
    done
  done
done
EOF

  chmod +x "$worker_script"
  setsid bash "$worker_script" > "$log_file" 2>&1 < /dev/null &
  echo "$!" > "$worker_dir/pid.txt"
  echo "Started worker $worker_id on GPU $gpu: $worker_script"
  echo "  log: $log_file"
done

echo "Run root: $RUN_ROOT"
