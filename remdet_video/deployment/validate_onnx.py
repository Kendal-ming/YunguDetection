"""Validate RemDet preprocessing and ONNX outputs against MMDetection."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from mmdet.apis import DetInferencer

from remdet_video.deployment.export_onnx import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFERENCE,
    sha256,
)
from remdet_video.deployment.model import RemDetONNXWrapper
from remdet_video.deployment.postprocess import (
    VISDRONE_CLASSES,
    box_iou,
    postprocess_detections,
)
from remdet_video.deployment.preprocess import preprocess_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--reference-image', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--onnx', type=Path, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-size', type=int, default=640)
    return parser.parse_args()


def summarize_error(reference: np.ndarray, candidate: np.ndarray) -> dict:
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    denominator = np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
    return {
        'shape': list(reference.shape),
        'max_abs': float(difference.max()),
        'mean_abs': float(difference.mean()),
        'p99_abs': float(np.quantile(difference, 0.99)),
        'max_relative': float((difference / denominator).max()),
        'allclose_rtol_1e-3_atol_1e-3': bool(np.allclose(
            reference, candidate, rtol=1e-3, atol=1e-3)),
    }


def instances_to_dicts(instances) -> list[dict]:
    instances = instances.cpu()
    return [{
        'class_id': int(label),
        'class_name': VISDRONE_CLASSES[int(label)],
        'score': float(score),
        'bbox': [float(value) for value in bbox],
    } for label, score, bbox in zip(
        instances.labels.numpy(), instances.scores.numpy(), instances.bboxes.numpy())]


def compare_detections(reference: list[dict], candidate: list[dict]) -> dict:
    """Greedily match same-class detections and summarize geometric error."""
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

    matched_ious = [item['iou'] for item in matches]
    score_errors = [item['score_abs_error'] for item in matches]
    return {
        'reference_count': len(reference),
        'candidate_count': len(candidate),
        'reference_class_counts': dict(Counter(
            item['class_name'] for item in reference)),
        'candidate_class_counts': dict(Counter(
            item['class_name'] for item in candidate)),
        'matched_count': len(matches),
        'matches_iou_ge_0_99': sum(iou >= 0.99 for iou in matched_ious),
        'minimum_matched_iou': min(matched_ious, default=None),
        'mean_matched_iou': float(np.mean(matched_ious)) if matched_ious else None,
        'maximum_score_abs_error': max(score_errors, default=None),
    }


def filter_visible(detections: list[dict], threshold: float = 0.25) -> list[dict]:
    return [item for item in detections if item['score'] >= threshold]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    onnx_path = (args.onnx.resolve() if args.onnx else
                 output_dir / f'remdet_s_{args.input_size}_fp32.onnx')
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    graph_contract = {
        'inputs': [value.name for value in onnx_model.graph.input],
        'outputs': [value.name for value in onnx_model.graph.output],
        'node_count': len(onnx_model.graph.node),
        'initializer_count': len(onnx_model.graph.initializer),
    }

    image = cv2.imread(str(args.reference_image.resolve()))
    if image is None:
        raise RuntimeError(f'OpenCV could not read {args.reference_image}')
    own_input, meta = preprocess_bgr(image, args.input_size)

    inferencer = DetInferencer(
        model=str(args.config.resolve()),
        weights=str(args.checkpoint.resolve()),
        device=args.device,
        palette='random',
        show_progress=False,
    )
    inferencer.model.dataset_meta['classes'] = VISDRONE_CLASSES
    inferencer.model.eval()

    _, model_batch = next(iter(inferencer.preprocess([image], batch_size=1)))
    model_data = inferencer.model.data_preprocessor(model_batch, training=False)
    framework_input = model_data['inputs']
    preprocess_error = summarize_error(
        framework_input.detach().cpu().numpy(), own_input)

    wrapper = RemDetONNXWrapper(
        inferencer.model,
        input_size=args.input_size,
        strides=inferencer.model.bbox_head.head_module.featmap_strides,
    ).eval().to(args.device)
    with torch.inference_mode():
        torch_boxes, torch_scores = wrapper(
            torch.from_numpy(own_input).to(args.device))
    torch_boxes_np = torch_boxes.detach().cpu().numpy()
    torch_scores_np = torch_scores.detach().cpu().numpy()

    session = ort.InferenceSession(
        str(onnx_path), providers=['CPUExecutionProvider'])
    ort_boxes, ort_scores = session.run(
        ['boxes', 'scores'], {'images': own_input})
    repeat_boxes, repeat_scores = session.run(
        ['boxes', 'scores'], {'images': own_input})
    determinism = {
        'boxes_max_abs': float(np.max(np.abs(ort_boxes - repeat_boxes))),
        'scores_max_abs': float(np.max(np.abs(ort_scores - repeat_scores))),
    }
    export_device_comparison = {
        'boxes': summarize_error(torch_boxes_np, ort_boxes),
        'scores': summarize_error(torch_scores_np, ort_scores),
    }

    # Recreate a fresh input batch because framework preprocessing may mutate it.
    _, original_batch = next(iter(inferencer.preprocess([image], batch_size=1)))
    with torch.inference_mode():
        original_predictions = inferencer.forward(original_batch)
    original_detections = instances_to_dicts(
        original_predictions[0].pred_instances)

    # ONNX Runtime uses CPU here. A second PyTorch CPU pass distinguishes
    # graph-conversion differences from ordinary CUDA-vs-CPU FP32 variation.
    wrapper = wrapper.cpu()
    with torch.inference_mode():
        cpu_boxes, cpu_scores = wrapper(torch.from_numpy(own_input))
    cpu_comparison = {
        'boxes': summarize_error(cpu_boxes.numpy(), ort_boxes),
        'scores': summarize_error(cpu_scores.numpy(), ort_scores),
    }
    raw_comparison = {
        'torch_export_device_vs_onnx_cpu': export_device_comparison,
        'torch_cpu_vs_onnx_cpu': cpu_comparison,
    }
    torch_detections = postprocess_detections(
        torch_boxes_np, torch_scores_np, meta)
    ort_detections = postprocess_detections(ort_boxes, ort_scores, meta)

    full_comparison = {
        'mmdet_vs_torch_wrapper': compare_detections(
            original_detections, torch_detections),
        'torch_wrapper_vs_onnx': compare_detections(
            torch_detections, ort_detections),
    }
    visible_comparison = {
        'threshold': 0.25,
        'mmdet_vs_torch_wrapper': compare_detections(
            filter_visible(original_detections),
            filter_visible(torch_detections)),
        'torch_wrapper_vs_onnx': compare_detections(
            filter_visible(torch_detections),
            filter_visible(ort_detections)),
    }

    full_mmdet_wrapper = full_comparison['mmdet_vs_torch_wrapper']
    full_wrapper_onnx = full_comparison['torch_wrapper_vs_onnx']
    visible_mmdet_wrapper = visible_comparison['mmdet_vs_torch_wrapper']
    visible_wrapper_onnx = visible_comparison['torch_wrapper_vs_onnx']
    box_error = export_device_comparison['boxes']
    score_error = export_device_comparison['scores']

    checks = {
        'onnx_checker_passed': True,
        'graph_names_correct': graph_contract['inputs'] == ['images'] and
                               graph_contract['outputs'] == ['boxes', 'scores'],
        'preprocess_exact': preprocess_error['max_abs'] == 0.0,
        # The decoded boxes are in 640-pixel input coordinates. These bounds
        # accept sub-quarter-pixel backend variation while rejecting a real
        # grid/stride/decode error, which would be several pixels or larger.
        'boxes_within_fp32_backend_tolerance': (
            box_error['max_abs'] <= 0.25 and box_error['mean_abs'] <= 0.01),
        'scores_within_fp32_backend_tolerance': (
            score_error['max_abs'] <= 0.001 and score_error['mean_abs'] <= 1e-5),
        'finite_outputs': bool(
            np.isfinite(ort_boxes).all() and np.isfinite(ort_scores).all()),
        'onnxruntime_repeat_deterministic': (
            determinism['boxes_max_abs'] == 0.0 and
            determinism['scores_max_abs'] == 0.0),
        'full_detection_counts_match': (
            full_mmdet_wrapper['reference_count'] ==
            full_mmdet_wrapper['candidate_count'] ==
            full_wrapper_onnx['candidate_count']),
        'full_onnx_matches_iou_0_99': (
            full_wrapper_onnx['matched_count'] ==
            full_wrapper_onnx['matches_iou_ge_0_99']),
        'visible_detection_counts_match': (
            visible_mmdet_wrapper['reference_count'] ==
            visible_mmdet_wrapper['candidate_count'] ==
            visible_wrapper_onnx['candidate_count']),
        'visible_onnx_matches_iou_0_99': (
            visible_wrapper_onnx['matched_count'] ==
            visible_wrapper_onnx['matches_iou_ge_0_99']),
    }
    passed = all(checks.values())

    input_path = output_dir / 'reference_input_fp32.npy'
    expected_outputs_path = output_dir / 'reference_expected_outputs.npz'
    detections_path = output_dir / 'reference_detections.json'
    np.save(input_path, own_input)
    np.savez_compressed(
        expected_outputs_path, boxes=ort_boxes, scores=ort_scores)
    detections_path.write_text(json.dumps({
        'mmdet': original_detections,
        'torch_wrapper': torch_detections,
        'onnx': ort_detections,
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'passed': passed,
        'checks': checks,
        'onnx': {
            'path': str(onnx_path),
            'sha256': sha256(onnx_path),
            'size_bytes': onnx_path.stat().st_size,
            'runtime_version': ort.__version__,
            'providers': session.get_providers(),
        },
        'reference_image': str(args.reference_image.resolve()),
        'preprocess_meta': meta.to_dict(),
        'preprocess_comparison': preprocess_error,
        'graph_contract': graph_contract,
        'raw_output_comparison': raw_comparison,
        'onnxruntime_determinism': determinism,
        'postprocess_full_threshold_0_001': full_comparison,
        'postprocess_visible_threshold_0_25': visible_comparison,
        'reference_artifacts': {
            'input_npy': {
                'path': str(input_path),
                'sha256': sha256(input_path),
                'shape': list(own_input.shape),
            },
            'expected_outputs_npz': {
                'path': str(expected_outputs_path),
                'sha256': sha256(expected_outputs_path),
            },
            'detections_json': {
                'path': str(detections_path),
                'sha256': sha256(detections_path),
            },
        },
        'sample_visible_detections': {
            'mmdet': filter_visible(original_detections)[:20],
            'torch_wrapper': filter_visible(torch_detections)[:20],
            'onnx': filter_visible(ort_detections)[:20],
        },
    }
    report_path = output_dir / 'validation_report.json'
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    manifest_path = output_dir / 'deployment_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['validation'] = {
            'passed': passed,
            'report_path': str(report_path),
            'report_sha256': sha256(report_path),
            'reference_input_path': str(input_path),
            'reference_input_sha256': sha256(input_path),
            'expected_outputs_path': str(expected_outputs_path),
            'expected_outputs_sha256': sha256(expected_outputs_path),
            'detections_path': str(detections_path),
            'detections_sha256': sha256(detections_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'passed': passed,
        'checks': checks,
        'preprocess_max_abs': preprocess_error['max_abs'],
        'raw_output_comparison': raw_comparison,
        'visible_comparison': visible_comparison,
        'report': str(report_path),
    }, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
