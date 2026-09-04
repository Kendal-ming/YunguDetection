#!/usr/bin/env bash
set -Eeuo pipefail

DEPLOY_ROOT="/home/nvidia/remdet_deploy"
MODEL="${DEPLOY_ROOT}/models/remdet_s_640_fp32.onnx"
ENGINES_DIR="${DEPLOY_ROOT}/engines"
RESULTS_DIR="${DEPLOY_ROOT}/results"
FP32_ENGINE="${ENGINES_DIR}/remdet_s_640_fp32.engine"
FP16_ENGINE="${ENGINES_DIR}/remdet_s_640_fp16.engine"
RESULT_BUNDLE="${DEPLOY_ROOT}/remdet_all_results.tar.gz"
CONSOLE_LOG="${RESULTS_DIR}/true_15w_rebuild_console.log"

mkdir -p "${ENGINES_DIR}" "${RESULTS_DIR}"
exec > >(tee "${CONSOLE_LOG}") 2>&1

package_partial_results() {
  local status=$?
  echo
  echo "Packaging all available results (exit status: ${status})..."
  tar -czf "${RESULT_BUNDLE}" -C "${RESULTS_DIR}" . || true
  if [[ -f "${RESULT_BUNDLE}" ]]; then
    sha256sum "${RESULT_BUNDLE}" || true
    ls -lh "${RESULT_BUNDLE}" || true
  fi
  return "${status}"
}
trap package_partial_results EXIT

echo "============================================================"
echo "RemDet true-15W engine rebuild and complete validation"
date --iso-8601=seconds
echo "============================================================"

if [[ ! -f "${MODEL}" ]]; then
  echo "ERROR: ONNX model is missing: ${MODEL}" >&2
  exit 2
fi

power_mode=$(nvpmodel -q 2>&1 || true)
tpc_mask=$(cat /sys/devices/platform/gpu.0/tpc_pg_mask)
echo "Power mode query:"
echo "${power_mode}"
echo "TPC power-gating mask: ${tpc_mask}"
if [[ "${power_mode}" != *"15W"* || "${tpc_mask}" != "252" ]]; then
  echo "ERROR: expected the rebooted 15W configuration with TPC mask 252." >&2
  exit 2
fi

archive_dir="${ENGINES_DIR}/archive_8sm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${archive_dir}"
for engine in "${FP32_ENGINE}" "${FP16_ENGINE}"; do
  if [[ -f "${engine}" ]]; then
    mv "${engine}" "${archive_dir}/"
  fi
done
echo "Previous 8-SM engines preserved in: ${archive_dir}"

echo
echo "[1/7] Building FP32 engine for the active 4-SM 15W configuration..."
trtexec \
  --onnx="${MODEL}" \
  --saveEngine="${FP32_ENGINE}" \
  --noTF32 \
  --memPoolSize=workspace:4096 \
  --skipInference 2>&1 | tee "${RESULTS_DIR}/build_fp32_true_15w.log"

echo
echo "[2/7] Building FP16 engine for the active 4-SM 15W configuration..."
trtexec \
  --onnx="${MODEL}" \
  --saveEngine="${FP16_ENGINE}" \
  --fp16 \
  --noTF32 \
  --memPoolSize=workspace:4096 \
  --skipInference 2>&1 | tee "${RESULTS_DIR}/build_fp16_true_15w.log"

echo
echo "New engine hashes:"
sha256sum "${FP32_ENGINE}" "${FP16_ENGINE}"
ls -lh "${FP32_ENGINE}" "${FP16_ENGINE}"

echo
echo "[3/7] Short FP32 engine smoke test..."
trtexec \
  --loadEngine="${FP32_ENGINE}" \
  --warmUp=100 \
  --iterations=10 \
  --duration=0 \
  --noDataTransfers 2>&1 | tee "${RESULTS_DIR}/smoke_fp32_true_15w.log"

echo
echo "[4/7] Short FP16 engine smoke test..."
trtexec \
  --loadEngine="${FP16_ENGINE}" \
  --warmUp=100 \
  --iterations=10 \
  --duration=0 \
  --noDataTransfers 2>&1 | tee "${RESULTS_DIR}/smoke_fp16_true_15w.log"

echo
echo "[5/7] Numerical validation of FP32 and FP16 engines..."
python3 "${DEPLOY_ROOT}/scripts/validate_trt.py" \
  --precision fp32 \
  --engine "${FP32_ENGINE}" \
  --output "${RESULTS_DIR}/trt_fp32_validation.json"
python3 "${DEPLOY_ROOT}/scripts/validate_trt.py" \
  --precision fp16 \
  --engine "${FP16_ENGINE}" \
  --output "${RESULTS_DIR}/trt_fp16_validation.json"

echo
echo "[6/7] Re-running the pure-engine FP32/FP16 15W benchmark..."
python3 "${DEPLOY_ROOT}/scripts/benchmark_engines.py"

echo
echo "[7/7] Running video and full VisDrone validation-set inference..."
bash "${DEPLOY_ROOT}/scripts/run_remaining_tests.sh"

echo
echo "True-15W rebuild and every remaining Jetson test completed."
echo "Result bundle: ${RESULT_BUNDLE}"
