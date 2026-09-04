"""Separate GPU network stages from box decoding and NMS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remdet_video.core.detector import RemDetDetector  # noqa: E402
from remdet_video.evaluation.statistics import summarize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=500)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    detector = RemDetDetector(args.config, args.checkpoint, deploy=False)
    capture = cv2.VideoCapture(str(Path(args.video).resolve()))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f'Cannot read {args.video}')

    _, model_inputs = next(iter(
        detector.inferencer.preprocess([frame], batch_size=1)))
    processed = detector.model.data_preprocessor(model_inputs, training=False)
    batch_inputs = processed['inputs']
    batch_data_samples = processed['data_samples']
    batch_img_metas = [sample.metainfo for sample in batch_data_samples]

    def forward_once():
        features = detector.model.extract_feat(batch_inputs)
        head_outputs = detector.model.bbox_head(features)
        results = detector.model.bbox_head.predict_by_feat(
            *head_outputs,
            batch_img_metas=batch_img_metas,
            rescale=True,
            with_nms=True)
        return results

    for _ in range(args.warmup):
        forward_once()
    torch.cuda.synchronize()

    samples = {
        'backbone_neck_gpu_ms': [],
        'head_gpu_ms': [],
        'decode_nms_gpu_ms': [],
        'raw_gpu_total_ms': [],
        'raw_wall_total_ms': [],
    }
    detection_count = 0
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        feature_end = torch.cuda.Event(enable_timing=True)
        head_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = perf_counter()
        start.record()
        features = detector.model.extract_feat(batch_inputs)
        feature_end.record()
        head_outputs = detector.model.bbox_head(features)
        head_end.record()
        results = detector.model.bbox_head.predict_by_feat(
            *head_outputs,
            batch_img_metas=batch_img_metas,
            rescale=True,
            with_nms=True)
        end.record()
        torch.cuda.synchronize()
        samples['backbone_neck_gpu_ms'].append(
            start.elapsed_time(feature_end))
        samples['head_gpu_ms'].append(feature_end.elapsed_time(head_end))
        samples['decode_nms_gpu_ms'].append(head_end.elapsed_time(end))
        samples['raw_gpu_total_ms'].append(start.elapsed_time(end))
        samples['raw_wall_total_ms'].append(
            (perf_counter() - wall_start) * 1000.0)
        detection_count = len(results[0])

    summary = {
        'experiment': args.experiment,
        'iterations': args.iterations,
        'input_shape': list(batch_inputs.shape),
        'detections_after_nms': detection_count,
        'parameter_count': detector.parameter_count,
        'timing': {name: summarize(values) for name, values in samples.items()},
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
