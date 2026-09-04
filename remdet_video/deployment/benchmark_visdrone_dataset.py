"""Run RemDet TensorRT FP16 over the full VisDrone2019-DET validation set.

The Jetson writes standard COCO detection results for evaluation on Windows
and separately records end-to-end latency plus tegrastats telemetry. Evaluation
is intentionally kept off the Jetson so it does not need MMDetection or
pycocotools.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

from benchmark_image_pipeline import (
    distribution,
    parse_tegrastats,
    read_power_mode,
    sha256,
)
from postprocess import VISDRONE_CLASSES, postprocess_detections
from preprocess import preprocess_bgr
from trt_runtime import TensorRTRunner


DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--engine', type=Path,
        default=DEPLOY_ROOT / 'engines/remdet_s_640_fp16.engine')
    parser.add_argument(
        '--images', type=Path,
        default=DEPLOY_ROOT / 'datasets/visdrone_val/images')
    parser.add_argument(
        '--annotations', type=Path,
        default=DEPLOY_ROOT /
        'datasets/visdrone_val/VisDrone2019-DET_val_coco.json')
    parser.add_argument('--score-threshold', type=float, default=0.001)
    parser.add_argument('--iou-threshold', type=float, default=0.7)
    parser.add_argument('--max-detections', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--telemetry-interval-ms', type=int, default=200)
    parser.add_argument('--progress-interval', type=int, default=50)
    parser.add_argument(
        '--predictions', type=Path,
        default=DEPLOY_ROOT /
        'results/visdrone_val_predictions_trt_fp16.json')
    parser.add_argument(
        '--output', type=Path,
        default=DEPLOY_ROOT /
        'results/visdrone_val_inference_trt_fp16_15w.json')
    return parser.parse_args()


def coco_prediction(image_id: int, detection: dict) -> dict:
    x1, y1, x2, y2 = detection['bbox']
    return {
        'image_id': image_id,
        'category_id': int(detection['class_id']) + 1,
        'bbox': [
            float(x1),
            float(y1),
            float(max(0.0, x2 - x1)),
            float(max(0.0, y2 - y1)),
        ],
        'score': float(detection['score']),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.max_detections <= 0:
        raise ValueError('Warmup must be non-negative and max detections positive.')
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError('Score threshold must be between 0 and 1.')
    for path in (args.engine, args.images, args.annotations):
        if not path.exists():
            raise FileNotFoundError(path)

    annotation_data = json.loads(
        args.annotations.read_text(encoding='utf-8'))
    images = sorted(annotation_data['images'], key=lambda item: item['id'])
    categories = sorted(
        annotation_data['categories'], key=lambda item: item['id'])
    category_names = tuple(item['name'] for item in categories)
    expected_ids = list(range(1, len(VISDRONE_CLASSES) + 1))
    actual_ids = [int(item['id']) for item in categories]
    if actual_ids != expected_ids or category_names != VISDRONE_CLASSES:
        raise RuntimeError(
            'Dataset categories do not match the exported RemDet class order: '
            f'{categories}')
    if not images:
        raise RuntimeError('The annotation file contains no images.')

    missing = [
        str(args.images / item['file_name'])
        for item in images
        if not (args.images / item['file_name']).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} validation images are missing; first: {missing[0]}')

    sample = cv2.imread(str(args.images / images[0]['file_name']))
    if sample is None:
        raise RuntimeError('OpenCV could not decode the first validation image.')

    print(
        f'Loading FP16 engine and preparing {len(images)} validation images...',
        flush=True)
    load_started = time.perf_counter_ns()
    runner = TensorRTRunner(args.engine)
    engine_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000.0

    all_finite = True
    try:
        print(f'Warmup: {args.warmup} runs', flush=True)
        warmup_tensor, _ = preprocess_bgr(sample, 640)
        for _ in range(args.warmup):
            warmup_outputs = runner.run({'images': warmup_tensor})
            all_finite = all_finite and bool(
                np.isfinite(warmup_outputs['boxes']).all() and
                np.isfinite(warmup_outputs['scores']).all())

        telemetry_path = args.output.parent / 'tegrastats_visdrone_val.log'
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path.unlink(missing_ok=True)
        telemetry = subprocess.Popen(
            ['tegrastats', '--interval', str(args.telemetry_interval_ms),
             '--logfile', str(telemetry_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(0.5)

        timings: dict[str, list[float]] = {
            'decode_ms': [],
            'preprocess_ms': [],
            'inference_with_transfers_ms': [],
            'postprocess_nms_ms': [],
            'end_to_end_ms': [],
        }
        predictions: list[dict] = []
        detection_class_counts: Counter[str] = Counter()
        per_image_detection_counts: list[int] = []
        started_all = time.perf_counter()
        try:
            for index, image_info in enumerate(images, start=1):
                start = time.perf_counter_ns()
                image_path = args.images / image_info['file_name']
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                after_decode = time.perf_counter_ns()
                if image is None:
                    raise RuntimeError(f'OpenCV could not read {image_path}')
                if (image.shape[1], image.shape[0]) != (
                        int(image_info['width']), int(image_info['height'])):
                    raise RuntimeError(
                        f'Image dimensions disagree with COCO metadata: '
                        f'{image_path}')

                tensor, meta = preprocess_bgr(image, 640)
                after_preprocess = time.perf_counter_ns()
                outputs = runner.run({'images': tensor})
                after_inference = time.perf_counter_ns()
                finite = bool(
                    np.isfinite(outputs['boxes']).all() and
                    np.isfinite(outputs['scores']).all())
                all_finite = all_finite and finite
                detections = postprocess_detections(
                    outputs['boxes'], outputs['scores'], meta,
                    score_threshold=args.score_threshold,
                    iou_threshold=args.iou_threshold,
                    max_detections=args.max_detections)
                after_postprocess = time.perf_counter_ns()

                to_ms = 1.0 / 1_000_000.0
                values = {
                    'decode_ms': (after_decode - start) * to_ms,
                    'preprocess_ms': (
                        after_preprocess - after_decode) * to_ms,
                    'inference_with_transfers_ms': (
                        after_inference - after_preprocess) * to_ms,
                    'postprocess_nms_ms': (
                        after_postprocess - after_inference) * to_ms,
                    'end_to_end_ms': (
                        after_postprocess - start) * to_ms,
                }
                for name, value in values.items():
                    timings[name].append(value)

                image_id = int(image_info['id'])
                predictions.extend(
                    coco_prediction(image_id, detection)
                    for detection in detections)
                per_image_detection_counts.append(len(detections))
                detection_class_counts.update(
                    item['class_name'] for item in detections)

                if (index % args.progress_interval == 0 or
                        index == len(images)):
                    elapsed = time.perf_counter() - started_all
                    print(
                        f'[{index}/{len(images)}] elapsed={elapsed:.1f}s, '
                        f'current_detections={len(detections)}, '
                        f'total_predictions={len(predictions)}', flush=True)
                    write_json(args.predictions, predictions)
        finally:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait(timeout=5)
    finally:
        runner.close()

    write_json(args.predictions, predictions)
    timing_summary = {
        name: distribution(values) for name, values in timings.items()
    }
    timing_summary['effective_fps_from_mean_e2e'] = (
        1000.0 / timing_summary['end_to_end_ms']['mean'])
    checks = {
        'processed_all_images': len(timings['end_to_end_ms']) == len(images),
        'finite_outputs': all_finite,
        'predictions_written': args.predictions.is_file(),
        'category_contract_correct': True,
    }
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'passed': all(checks.values()),
        'checks': checks,
        'benchmark': (
            'full VisDrone2019-DET val inference; COCO mAP evaluated on Windows'),
        'configuration': {
            'input_shape': [1, 3, 640, 640],
            'batch_size': 1,
            'score_threshold': args.score_threshold,
            'iou_threshold': args.iou_threshold,
            'max_detections_per_image': args.max_detections,
            'warmup_runs': args.warmup,
            'power_mode_query': read_power_mode(),
            'telemetry_interval_ms': args.telemetry_interval_ms,
        },
        'environment': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'tensorrt': trt.__version__,
            'opencv': cv2.__version__,
            'numpy': np.__version__,
        },
        'files': {
            'engine': str(args.engine.resolve()),
            'engine_sha256': sha256(args.engine),
            'annotations': str(args.annotations.resolve()),
            'annotations_sha256': sha256(args.annotations),
            'images_directory': str(args.images.resolve()),
            'predictions': str(args.predictions.resolve()),
            'predictions_sha256': sha256(args.predictions),
            'telemetry_log': str(telemetry_path.resolve()),
        },
        'dataset': {
            'image_count': len(images),
            'annotation_count': len(annotation_data['annotations']),
            'categories': categories,
        },
        'startup_engine_load_ms': engine_load_ms,
        'timings': timing_summary,
        'telemetry': parse_tegrastats(telemetry_path),
        'detections': {
            'total_predictions': len(predictions),
            'per_image_count': distribution(
                [float(value) for value in per_image_detection_counts]),
            'class_counts': dict(detection_class_counts),
        },
    }
    write_json(args.output, report)
    print(json.dumps({
        'passed': report['passed'],
        'checks': checks,
        'images': len(images),
        'total_predictions': len(predictions),
        'timings': timing_summary,
        'telemetry': report['telemetry'],
        'predictions': str(args.predictions.resolve()),
        'report': str(args.output.resolve()),
    }, indent=2, ensure_ascii=False), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
