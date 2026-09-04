"""Validate a Jetson TensorRT engine against the Windows ONNX reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorrt as trt

from postprocess import box_iou, postprocess_detections
from preprocess import LetterboxMeta
from trt_runtime import TensorRTRunner


DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--engine', type=Path,
        default=DEPLOY_ROOT / 'engines/remdet_s_640_fp32.engine')
    parser.add_argument(
        '--input', type=Path,
        default=DEPLOY_ROOT / 'data/reference_input_fp32.npy')
    parser.add_argument(
        '--expected', type=Path,
        default=DEPLOY_ROOT / 'data/reference_expected_outputs.npz')
    parser.add_argument(
        '--reference-detections', type=Path,
        default=DEPLOY_ROOT / 'data/reference_detections.json')
    parser.add_argument(
        '--manifest', type=Path,
        default=DEPLOY_ROOT / 'models/deployment_manifest.json')
    parser.add_argument(
        '--output', type=Path,
        default=DEPLOY_ROOT / 'results/trt_fp32_validation.json')
    parser.add_argument(
        '--precision', choices=('fp32', 'fp16'), default='fp32',
        help='Select numerical tolerances appropriate for the engine.')
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize_error(reference: np.ndarray, candidate: np.ndarray) -> dict:
    difference = np.abs(reference.astype(np.float64) -
                        candidate.astype(np.float64))
    return {
        'shape': list(reference.shape),
        'max_abs': float(difference.max()),
        'mean_abs': float(difference.mean()),
        'p99_abs': float(np.quantile(difference, 0.99)),
    }


def compare_detections(reference: list[dict], candidate: list[dict]) -> dict:
    unused = set(range(len(candidate)))
    matches = []
    for ref in reference:
        compatible = [index for index in unused
                      if candidate[index]['class_id'] == ref['class_id']]
        if not compatible:
            continue
        ref_box = np.asarray(ref['bbox'], dtype=np.float32)
        candidate_boxes = np.asarray(
            [candidate[index]['bbox'] for index in compatible], dtype=np.float32)
        ious = box_iou(ref_box, candidate_boxes)
        best_position = int(np.argmax(ious))
        best_index = compatible[best_position]
        unused.remove(best_index)
        matches.append({
            'iou': float(ious[best_position]),
            'score_abs_error': abs(ref['score'] - candidate[best_index]['score']),
        })
    ious = [match['iou'] for match in matches]
    score_errors = [match['score_abs_error'] for match in matches]
    return {
        'reference_count': len(reference),
        'candidate_count': len(candidate),
        'reference_class_counts': dict(Counter(
            item['class_name'] for item in reference)),
        'candidate_class_counts': dict(Counter(
            item['class_name'] for item in candidate)),
        'matched_count': len(matches),
        'matches_iou_ge_0_99': sum(iou >= 0.99 for iou in ious),
        'minimum_matched_iou': min(ious, default=None),
        'mean_matched_iou': float(np.mean(ious)) if ious else None,
        'maximum_score_abs_error': max(score_errors, default=None),
    }


def filter_visible(detections: list[dict]) -> list[dict]:
    return [item for item in detections if item['score'] >= 0.25]


def maximum_class_count_delta(comparison: dict) -> int:
    reference = comparison['reference_class_counts']
    candidate = comparison['candidate_class_counts']
    names = set(reference) | set(candidate)
    return max(
        (abs(reference.get(name, 0) - candidate.get(name, 0))
         for name in names),
        default=0,
    )


def main() -> None:
    args = parse_args()
    required = (
        args.engine,
        args.input,
        args.expected,
        args.reference_detections,
        args.manifest,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    input_array = np.load(args.input)
    with np.load(args.expected) as expected:
        expected_boxes = expected['boxes']
        expected_scores = expected['scores']
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    meta_dict = manifest['preprocess']['reference_meta']
    meta = LetterboxMeta(
        original_shape=tuple(meta_dict['original_shape']),
        resized_shape=tuple(meta_dict['resized_shape']),
        input_shape=tuple(meta_dict['input_shape']),
        scale_factor=tuple(meta_dict['scale_factor']),
        pad_param=tuple(meta_dict['pad_param']),
    )
    reference_payload = json.loads(
        args.reference_detections.read_text(encoding='utf-8'))
    reference_detections = reference_payload['onnx']

    with TensorRTRunner(args.engine) as runner:
        first = runner.run({'images': input_array})
        second = runner.run({'images': input_array})
        io_contract = runner.io_contract()
    if set(first) != {'boxes', 'scores'}:
        raise RuntimeError(f'Unexpected TensorRT outputs: {sorted(first)}')
    boxes = first['boxes']
    scores = first['scores']

    box_error = summarize_error(expected_boxes, boxes)
    score_error = summarize_error(expected_scores, scores)
    repeat_error = {
        'boxes_max_abs': float(np.max(np.abs(boxes - second['boxes']))),
        'scores_max_abs': float(np.max(np.abs(scores - second['scores']))),
    }
    detections = postprocess_detections(boxes, scores, meta)
    full_comparison = compare_detections(reference_detections, detections)
    visible_comparison = compare_detections(
        filter_visible(reference_detections), filter_visible(detections))

    if args.precision == 'fp32':
        tolerances = {
            'boxes_max_abs': 0.25,
            'boxes_mean_abs': 0.01,
            'scores_max_abs': 0.001,
            'scores_mean_abs': 1e-5,
            'minimum_detection_iou': 0.99,
            'mean_detection_iou': 0.99,
            'full_count_delta': 0,
            'visible_count_delta': 0,
            'class_count_delta': 0,
        }
    else:
        tolerances = {
            'boxes_max_abs': 1.0,
            'boxes_mean_abs': 0.05,
            'scores_max_abs': 0.01,
            'scores_mean_abs': 1e-4,
            'minimum_detection_iou': 0.98,
            'mean_detection_iou': 0.99,
            'full_count_delta': 0,
            'visible_count_delta': 1,
            'class_count_delta': 1,
        }

    full_minimum_iou = full_comparison['minimum_matched_iou']
    visible_minimum_iou = visible_comparison['minimum_matched_iou']
    if args.precision == 'fp32':
        full_iou_ok = (
            full_minimum_iou is not None and
            full_minimum_iou >= tolerances['minimum_detection_iou'])
        visible_iou_ok = (
            visible_minimum_iou is not None and
            visible_minimum_iou >= tolerances['minimum_detection_iou'])
    else:
        # Near-tied low-score FP16 detections can exchange order at the
        # max-detections boundary. Mean matched IoU is robust to that ordering.
        full_iou_ok = (
            full_comparison['mean_matched_iou'] is not None and
            full_comparison['mean_matched_iou'] >=
            tolerances['mean_detection_iou'])
        visible_iou_ok = (
            visible_comparison['mean_matched_iou'] is not None and
            visible_comparison['mean_matched_iou'] >=
            tolerances['mean_detection_iou'])

    checks = {
        'io_contract_correct': io_contract == {
            'inputs': {'images': {'shape': [1, 3, 640, 640], 'dtype': 'float32'}},
            'outputs': {
                'boxes': {'shape': [1, 8400, 4], 'dtype': 'float32'},
                'scores': {'shape': [1, 8400, 10], 'dtype': 'float32'},
            },
        },
        'finite_outputs': bool(
            np.isfinite(boxes).all() and np.isfinite(scores).all()),
        'boxes_within_precision_tolerance': (
            box_error['max_abs'] <= tolerances['boxes_max_abs'] and
            box_error['mean_abs'] <= tolerances['boxes_mean_abs']),
        'scores_within_precision_tolerance': (
            score_error['max_abs'] <= tolerances['scores_max_abs'] and
            score_error['mean_abs'] <= tolerances['scores_mean_abs']),
        'repeat_stable': (
            repeat_error['boxes_max_abs'] <= 1e-6 and
            repeat_error['scores_max_abs'] <= 1e-6),
        'full_detection_counts_within_tolerance': (
            abs(full_comparison['reference_count'] -
                full_comparison['candidate_count']) <=
            tolerances['full_count_delta'] and
            maximum_class_count_delta(full_comparison) <=
            tolerances['class_count_delta']),
        'full_detections_within_iou_tolerance': (
            full_comparison['matched_count'] >= min(
                full_comparison['reference_count'],
                full_comparison['candidate_count']) and full_iou_ok),
        'visible_detection_counts_within_tolerance': (
            abs(visible_comparison['reference_count'] -
                visible_comparison['candidate_count']) <=
            tolerances['visible_count_delta'] and
            maximum_class_count_delta(visible_comparison) <=
            tolerances['class_count_delta']),
        'visible_detections_within_iou_tolerance': (
            visible_comparison['matched_count'] >= min(
                visible_comparison['reference_count'],
                visible_comparison['candidate_count']) and visible_iou_ok),
    }
    passed = all(checks.values())
    model_path = Path('/proc/device-tree/model')
    device_model = (model_path.read_bytes().replace(b'\0', b'').decode('utf-8')
                    if model_path.is_file() else None)
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'passed': passed,
        'precision': args.precision,
        'tolerances': tolerances,
        'checks': checks,
        'environment': {
            'device': device_model,
            'tensorrt': trt.__version__,
            'numpy': np.__version__,
        },
        'engine': {
            'path': str(args.engine.resolve()),
            'sha256': sha256(args.engine),
            'size_bytes': args.engine.stat().st_size,
        },
        'io_contract': io_contract,
        'raw_output_error_vs_onnx': {
            'boxes': box_error,
            'scores': score_error,
        },
        'repeat_error': repeat_error,
        'postprocess_full_threshold_0_001': full_comparison,
        'postprocess_visible_threshold_0_25': visible_comparison,
        'sample_visible_detections': filter_visible(detections)[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'passed': passed,
        'checks': checks,
        'raw_output_error_vs_onnx': report['raw_output_error_vs_onnx'],
        'full_detection_comparison': full_comparison,
        'visible_detection_comparison': visible_comparison,
        'report': str(args.output.resolve()),
    }, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
