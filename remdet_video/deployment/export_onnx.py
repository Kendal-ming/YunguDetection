"""Export the fixed-shape RemDet-S 640 deployment graph to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mmcv
import mmengine
import onnx
import torch
from mmdet.apis import init_detector

from remdet_video.deployment.model import RemDetONNXWrapper
from remdet_video.deployment.postprocess import VISDRONE_CLASSES
from remdet_video.deployment.preprocess import preprocess_bgr


DEFAULT_CONFIG = PROJECT_ROOT / 'config_remdet/remdet/remdet_s-300e_visdrone.py'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'checkpoints/remdet_s_weights_only.pth'
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        'VISDRONE_DATA_ROOT',
        str(PROJECT_ROOT.parent / 'datasets/VisDrone2019-DET-COCO')))
DEFAULT_REFERENCE = (
    DEFAULT_DATA_ROOT / 'images/val/0000001_02999_d_0000005.jpg')
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'work_dirs/deployment/remdet_s_640'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--reference-image', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-size', type=int, default=640)
    parser.add_argument('--opset', type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    reference_image = args.reference_image.resolve()
    output_dir = args.output_dir.resolve()
    for required in (config, checkpoint, reference_image):
        if not required.is_file():
            raise FileNotFoundError(required)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(reference_image))
    if image is None:
        raise RuntimeError(f'OpenCV could not read {reference_image}')
    input_array, preprocess_meta = preprocess_bgr(image, args.input_size)

    model = init_detector(str(config), str(checkpoint), device=args.device)
    model.dataset_meta = {'classes': VISDRONE_CLASSES}
    model.eval()
    wrapper = RemDetONNXWrapper(
        model,
        input_size=args.input_size,
        strides=model.bbox_head.head_module.featmap_strides,
    ).eval().to(args.device)
    input_tensor = torch.from_numpy(input_array).to(args.device)

    with torch.inference_mode():
        boxes, scores = wrapper(input_tensor)
    expected_shapes = {
        'images': list(input_tensor.shape),
        'boxes': list(boxes.shape),
        'scores': list(scores.shape),
    }
    if expected_shapes != {
            'images': [1, 3, args.input_size, args.input_size],
            'boxes': [1, 8400, 4],
            'scores': [1, 8400, len(VISDRONE_CLASSES)],
    }:
        raise RuntimeError(f'Unexpected deployment shapes: {expected_shapes}')

    onnx_path = output_dir / f'remdet_s_{args.input_size}_fp32.onnx'
    torch.onnx.export(
        wrapper,
        (input_tensor,),
        str(onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['boxes', 'scores'],
        dynamic_axes=None,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    manifest = {
        'created_at': datetime.now().astimezone().isoformat(),
        'project': 'RemDet',
        'model_variant': 'RemDet-S',
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'precision': 'fp32',
        'fixed_batch_size': 1,
        'input_size': args.input_size,
        'opset': args.opset,
        'class_names': list(VISDRONE_CLASSES),
        'preprocess': {
            'source_color': 'BGR',
            'model_color': 'RGB',
            'resize': 'keep_ratio',
            'resize_interpolation_down': 'area',
            'resize_interpolation_up': 'bilinear',
            'letterbox_value': 114,
            'normalization': 'value / 255.0',
            'reference_meta': preprocess_meta.to_dict(),
        },
        'postprocess': {
            'multi_label': True,
            'score_threshold': 0.001,
            'nms_pre': 30000,
            'nms_iou_threshold': 0.7,
            'max_detections': 300,
            'nms_in_onnx': False,
        },
        'graph': {
            'input': {'name': 'images', 'shape': [1, 3, 640, 640], 'dtype': 'float32'},
            'outputs': {
                'boxes': {'shape': [1, 8400, 4], 'format': 'xyxy input pixels'},
                'scores': {'shape': [1, 8400, 10], 'activation': 'sigmoid'},
            },
            'repdwconv_fused': False,
            'dynamic_shapes': False,
        },
        'files': {
            'config': {'path': str(config), 'sha256': sha256(config)},
            'checkpoint': {'path': str(checkpoint), 'sha256': sha256(checkpoint)},
            'reference_image': {
                'path': str(reference_image),
                'sha256': sha256(reference_image),
            },
            'onnx': {
                'path': str(onnx_path),
                'sha256': sha256(onnx_path),
                'size_bytes': onnx_path.stat().st_size,
            },
        },
        'windows_baseline': {
            'bbox_mAP': 0.247,
            'bbox_mAP_50': 0.415,
            'bbox_mAP_75': 0.250,
            'end_to_end_mean_ms': 10.737513878848404,
            'end_to_end_fps': 93.1314279341588,
        },
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'mmcv': mmcv.__version__,
            'mmengine': mmengine.__version__,
            'onnx': onnx.__version__,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    manifest_path = output_dir / 'deployment_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'onnx': str(onnx_path),
        'manifest': str(manifest_path),
        'shapes': expected_shapes,
        'onnx_size_mb': onnx_path.stat().st_size / (1024 ** 2),
        'onnx_sha256': manifest['files']['onnx']['sha256'],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
