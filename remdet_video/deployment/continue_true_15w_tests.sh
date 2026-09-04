#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_ROOT="/home/nvidia/remdet_deploy"
RESULTS_DIR="${DEPLOY_ROOT}/results"
FP16_ENGINE="${DEPLOY_ROOT}/engines/remdet_s_640_fp16.engine"
CONSOLE_LOG="${RESULTS_DIR}/true_15w_continuation_console.log"

mkdir -p "${RESULTS_DIR}"
exec > >(tee "${CONSOLE_LOG}") 2>&1

echo "============================================================"
echo "Continue RemDet true-15W tests without rebuilding engines"
date --iso-8601=seconds
echo "============================================================"

echo "[1/4] Re-checking FP16 numerical output with boundary tolerance..."
python3 "${DEPLOY_ROOT}/scripts/validate_trt.py" \
  --precision fp16 \
  --engine "${FP16_ENGINE}" \
  --output "${RESULTS_DIR}/trt_fp16_validation.json"

echo
echo "[2/4] Measuring true-15W FP32/FP16 pure-engine performance..."
python3 "${DEPLOY_ROOT}/scripts/benchmark_engines.py"

echo
echo "[3/4] Measuring the repeated-image complete pipeline..."
python3 "${DEPLOY_ROOT}/scripts/benchmark_image_pipeline.py"

echo
echo "[4/4] Running video and all 548 VisDrone validation images..."
bash "${DEPLOY_ROOT}/scripts/run_remaining_tests.sh"

echo
echo "All true-15W tests completed."
echo "Result bundle: ${DEPLOY_ROOT}/remdet_all_results.tar.gz"
