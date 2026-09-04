#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT=/home/nvidia/remdet_deploy
ONNX_PATH="$DEPLOY_ROOT/models/remdet_s_640_fp32.onnx"
ENGINE_PATH="$DEPLOY_ROOT/engines/remdet_s_640_fp32.engine"
LOG_PATH="$DEPLOY_ROOT/results/build_fp32.log"
HASH_PATH="$DEPLOY_ROOT/results/remdet_s_640_fp32.engine.sha256"

if ! command -v trtexec >/dev/null 2>&1; then
    echo "ERROR: trtexec was not found in PATH." >&2
    exit 1
fi
if [[ ! -f "$ONNX_PATH" ]]; then
    echo "ERROR: ONNX model not found: $ONNX_PATH" >&2
    exit 1
fi

mkdir -p "$DEPLOY_ROOT/engines" "$DEPLOY_ROOT/results"
echo "Building strict FP32 TensorRT engine on this Jetson..."
echo "Input:  $ONNX_PATH"
echo "Output: $ENGINE_PATH"

trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --noTF32 \
    --memPoolSize=workspace:4096 \
    --skipInference \
    2>&1 | tee "$LOG_PATH"

sha256sum "$ENGINE_PATH" | tee "$HASH_PATH"
echo "FP32 engine build completed successfully."
