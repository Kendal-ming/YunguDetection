# RemDet-S 640 Jetson deployment bundle

This bundle is designed for `/home/nvidia/remdet_deploy` on the Jetson Orin
NX. It uses the TensorRT, CUDA, NumPy and OpenCV packages already supplied by
JetPack. It does not require PyTorch, MMCV, MMEngine or MMDetection.

The ONNX graph accepts FP32 `images` with shape `[1,3,640,640]` and returns
decoded `boxes` `[1,8400,4]` plus sigmoid `scores` `[1,8400,10]`. Thresholding
and class-aware NMS remain outside the engine.

Later stages use:

```bash
bash /home/nvidia/remdet_deploy/scripts/build_fp32.sh
python3 /home/nvidia/remdet_deploy/scripts/validate_trt.py
```

Do not run those commands until the transferred bundle and ONNX SHA-256 have
been checked.
