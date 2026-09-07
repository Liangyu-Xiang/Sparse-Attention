#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/runs/oracle_region_gpu012_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

declare -a GPUS=(0 1 2)
declare -a CONFIGS=(
  "eval_vggt_eth3d_oracle_2x2.yaml"
  "eval_vggt_eth3d_oracle_4x4.yaml"
  "eval_vggt_eth3d_oracle_8x8.yaml"
)

for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  config="${CONFIGS[$idx]}"
  log_path="${RUN_DIR}/gpu${gpu}_${config%.yaml}.log"
  echo "GPU ${gpu}: ${config} -> ${log_path}"
  (
    cd "${ROOT_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      /data/mmc_lyxiang/miniconda3/envs/saf3r_eval/bin/python evaluation/launch.py \
      --config_path ../configs/evaluation \
      --config "${config}"
  ) >"${log_path}" 2>&1
  echo "GPU ${gpu}: ${config} completed with status $?"
done
