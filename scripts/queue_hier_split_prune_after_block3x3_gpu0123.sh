#!/bin/bash
set -euo pipefail

PREV_RUN_ROOT="${PREV_RUN_ROOT:-runs/block3x3_violation_20260905_111702}"
RUN_ROOT="${RUN_ROOT:-runs/hier_split_prune_$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-saf3r_eval}"
CONFIG="${CONFIG:-eval_vggt_eth3d_hier_split_prune_guided.yaml}"
MAX_FRAMES="${MAX_FRAMES:-16}"
PATCH_BUDGETS="${PATCH_BUDGETS:-64}"
ROUNDS="${ROUNDS:-1 3 5 10}"
GAMMAS="${GAMMAS:-1.0}"
METHODS="${METHODS:-uniform topk random hierarchical}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-32}"
SAMPLE_MODE="${SAMPLE_MODE:-uniform}"
USE_MULTIPLICITY="${USE_MULTIPLICITY:-True}"
WAIT_FOR_PREVIOUS="${WAIT_FOR_PREVIOUS:-1}"
GPUS_CSV="${GPUS:-0,1,2,3}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "No GPUs configured." >&2
  exit 1
fi

SCENES=(
  courtyard electro kicker pipes relief delivery_area
  facade office playground relief_2 terrains
)

mkdir -p "$RUN_ROOT"
printf '%s\n' "${SCENES[@]}" > "$RUN_ROOT/all_scenes.txt"

wait_for_previous_run() {
  echo "Waiting for previous run: $PREV_RUN_ROOT"
  shopt -s nullglob
  pid_files=("$PREV_RUN_ROOT"/worker_*_gpu*/pid.txt)
  shopt -u nullglob
  if [ "${#pid_files[@]}" -eq 0 ]; then
    echo "No previous pid files found; continuing."
    return
  fi

  for pid_file in "${pid_files[@]}"; do
    pid="$(cat "$pid_file")"
    echo "Waiting for previous worker pid=$pid ($pid_file)"
    while ps -p "$pid" >/dev/null 2>&1; do
      sleep 60
    done
  done
  echo "Previous run workers have exited."
}

launch_workers() {
  for worker_id in "${!GPUS[@]}"; do
    gpu="${GPUS[$worker_id]}"
    worker_dir="$RUN_ROOT/worker_${worker_id}_gpu${gpu}"
    mkdir -p "$worker_dir/results" "$worker_dir/stats"

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
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
trap 'status=\$?; echo "exit_status=\$status"; echo "\$status" > "$worker_dir/exit_code.txt"' EXIT
PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)

echo "worker_id=$worker_id"
echo "gpu=$gpu"
echo "scenes=${scenes[*]}"
echo "config=$CONFIG"
echo "run_dir=$worker_dir"
echo "max_frames=$MAX_FRAMES patch_budgets=$PATCH_BUDGETS rounds=$ROUNDS gammas=$GAMMAS methods=$METHODS query_chunk_size=$QUERY_CHUNK_SIZE use_multiplicity=$USE_MULTIPLICITY"

for budget in $PATCH_BUDGETS; do
  for method in $METHODS; do
    if [ "\$method" = "uniform" ] || [ "\$method" = "topk" ]; then
      stats_dir="$worker_dir/stats/\${method}_B\${budget}"
      mkdir -p "\$stats_dir"
      "\${PYTHON_CMD[@]}" evaluation/launch.py --config "$CONFIG" \\
        "$scene_override" \\
        "datasets.eth3d.max_frames=$MAX_FRAMES" \\
        "datasets.eth3d.sample_mode=$SAMPLE_MODE" \\
        "model.patch_module.strategy=\$method" \\
        "model.patch_module.patch_budget=\$budget" \\
        "model.patch_module.max_rounds=0" \\
        "model.patch_module.use_multiplicity_correction=$USE_MULTIPLICITY" \\
        "model.patch_module.query_chunk_size=$QUERY_CHUNK_SIZE" \\
        "model.patch_module.stats_dir=\$stats_dir"
      continue
    fi

    for rounds in $ROUNDS; do
      if [ "\$method" = "random" ]; then
        stats_dir="$worker_dir/stats/random_B\${budget}_T\${rounds}"
        mkdir -p "\$stats_dir"
        "\${PYTHON_CMD[@]}" evaluation/launch.py --config "$CONFIG" \\
          "$scene_override" \\
          "datasets.eth3d.max_frames=$MAX_FRAMES" \\
          "datasets.eth3d.sample_mode=$SAMPLE_MODE" \\
          "model.patch_module.strategy=random" \\
          "model.patch_module.patch_budget=\$budget" \\
          "model.patch_module.max_rounds=\$rounds" \\
          "model.patch_module.use_multiplicity_correction=$USE_MULTIPLICITY" \\
          "model.patch_module.query_chunk_size=$QUERY_CHUNK_SIZE" \\
          "model.patch_module.stats_dir=\$stats_dir"
        continue
      fi

      for gamma in $GAMMAS; do
        stats_dir="$worker_dir/stats/hierarchical_B\${budget}_T\${rounds}_g\${gamma}"
        mkdir -p "\$stats_dir"
        "\${PYTHON_CMD[@]}" evaluation/launch.py --config "$CONFIG" \\
          "$scene_override" \\
          "datasets.eth3d.max_frames=$MAX_FRAMES" \\
          "datasets.eth3d.sample_mode=$SAMPLE_MODE" \\
          "model.patch_module.strategy=hierarchical" \\
          "model.patch_module.patch_budget=\$budget" \\
          "model.patch_module.max_rounds=\$rounds" \\
          "model.patch_module.gamma=\$gamma" \\
          "model.patch_module.use_multiplicity_correction=$USE_MULTIPLICITY" \\
          "model.patch_module.query_chunk_size=$QUERY_CHUNK_SIZE" \\
          "model.patch_module.stats_dir=\$stats_dir"
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
}

echo "Queue root: $RUN_ROOT"
echo "Started at: $(date -Is)"
echo "GPUs: ${GPUS[*]}"
if [ "$WAIT_FOR_PREVIOUS" = "1" ]; then
  wait_for_previous_run
else
  echo "Skipping wait for previous run."
fi
echo "Launching hierarchical split-prune workers at: $(date -Is)"
launch_workers
echo "Launched all workers."
