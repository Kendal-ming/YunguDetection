#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_ROOT="/home/nvidia/remdet_deploy"
RESULTS_DIR="${DEPLOY_ROOT}/results"
RESULT_BUNDLE="${DEPLOY_ROOT}/remdet_all_results.tar.gz"
CONSOLE_LOG="${RESULTS_DIR}/remaining_tests_console.log"

mkdir -p "${RESULTS_DIR}"
exec > >(tee "${CONSOLE_LOG}") 2>&1

package_results() {
  local status=$?
  echo
  echo "Packaging all available results (exit status: ${status})..."
  tar -czf "${RESULT_BUNDLE}" -C "${RESULTS_DIR}" . || true
  if [[ -f "${RESULT_BUNDLE}" ]]; then
    sha256sum "${RESULT_BUNDLE}" || true
    ls -lh "${RESULT_BUNDLE}" || true
  fi
  echo "Result bundle: ${RESULT_BUNDLE}"
  return "${status}"
}
trap package_results EXIT

echo "============================================================"
echo "RemDet remaining Jetson tests"
date --iso-8601=seconds
echo "============================================================"

required_files=(
  "${DEPLOY_ROOT}/engines/remdet_s_640_fp16.engine"
  "${DEPLOY_ROOT}/scripts/benchmark_video_pipeline.py"
  "${DEPLOY_ROOT}/scripts/benchmark_visdrone_dataset.py"
  "${DEPLOY_ROOT}/scripts/benchmark_image_pipeline.py"
  "${DEPLOY_ROOT}/scripts/preprocess.py"
  "${DEPLOY_ROOT}/scripts/postprocess.py"
  "${DEPLOY_ROOT}/scripts/trt_runtime.py"
  "${DEPLOY_ROOT}/data/demo_720p_h264.mp4"
  "${DEPLOY_ROOT}/datasets/visdrone_val/VisDrone2019-DET_val_coco.json"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required file is missing: ${path}" >&2
    exit 2
  fi
done

image_count=$(find "${DEPLOY_ROOT}/datasets/visdrone_val/images" \
  -maxdepth 1 -type f -iname '*.jpg' | wc -l)
if [[ "${image_count}" -ne 548 ]]; then
  echo "ERROR: expected 548 validation images, found ${image_count}." >&2
  exit 2
fi

echo "Preflight passed: engine, scripts, video and 548 images are present."

{
  echo "Collected at: $(date --iso-8601=seconds)"
  echo -n "Device: "
  tr -d '\0' < /proc/device-tree/model || true
  echo
  cat /etc/nv_tegra_release || true
  echo
  nvcc --version || true
  echo
  trtexec --version || true
  echo
  python3 -c "import sys, cv2, numpy, tensorrt; print('Python:', sys.version); print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('TensorRT:', tensorrt.__version__)"
  echo
  nvpmodel -q || true
  echo
  free -h || true
  df -h / || true
} > "${RESULTS_DIR}/environment_inventory.txt" 2>&1

echo
echo "[1/2] Running complete video-pipeline benchmark..."
python3 "${DEPLOY_ROOT}/scripts/benchmark_video_pipeline.py"

echo
echo "[2/2] Running all 548 VisDrone validation images..."
python3 "${DEPLOY_ROOT}/scripts/benchmark_visdrone_dataset.py"

echo
echo "All remaining Jetson tests completed successfully."
echo "Copy this file back to Windows: ${RESULT_BUNDLE}"
