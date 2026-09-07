#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/runs/oracle_2x2_remaining_max20_gpu012_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

declare -a GPUS=(0 1 2)
declare -a SCENES=(
  "[office,relief_2]"
  "[delivery_area,pipes,terrains]"
  "[facade,playground]"
)
CONFIG="eval_vggt_eth3d_oracle_2x2_all_layers_max20.yaml"

run_worker() {
  local gpu="$1"
  local scenes="$2"
  local worker_log="${RUN_DIR}/gpu${gpu}_worker.log"
  {
    echo "GPU ${gpu} assigned scenes ${scenes}"
    echo "[$(date -Is)] GPU ${gpu} starting ${CONFIG} (max_frames=20)"
    (
      cd "${ROOT_DIR}"
      CUDA_VISIBLE_DEVICES="${gpu}" \
        /data/mmc_lyxiang/miniconda3/envs/saf3r_eval/bin/python evaluation/launch.py \
        --config_path ../configs/evaluation \
        --config "${CONFIG}" \
        "datasets.eth3d.scenes=${scenes}"
    ) >"${RUN_DIR}/gpu${gpu}_${CONFIG%.yaml}.log" 2>&1
    echo "[$(date -Is)] GPU ${gpu} completed ${CONFIG}"
  } >"${worker_log}" 2>&1
}

for idx in "${!GPUS[@]}"; do
  run_worker "${GPUS[$idx]}" "${SCENES[$idx]}" &
  echo $! >"${RUN_DIR}/gpu${GPUS[$idx]}.pid"
done

wait
