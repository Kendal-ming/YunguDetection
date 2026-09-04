"""Benchmark RemDet on a video and cache low-threshold frame predictions."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import cv2
import mmcv
import mmengine
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
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--precision',
        choices=('fp32', 'amp-fp16'),
        default='fp32',
        help='FP32 or CUDA automatic mixed precision with FP16 compute.')
    parser.add_argument('--input-size', type=int, default=640)
    parser.add_argument('--raw-score-thr', type=float, default=0.001)
    parser.add_argument('--render-score-thr', type=float, default=0.30)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument(
        '--switch-to-deploy', action='store_true',
        help=('Try repository RepDWConv branch merging. The current upstream '
              'implementation is known to fail its equivalence smoke test.'))
    return parser.parse_args()


def draw_detections(frame, detections, threshold: float):
    start = perf_counter()
    output = frame.copy()
    for detection in detections:
        if detection.score < threshold:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 210, 40), 2)
        text = f'{detection.class_name} {detection.score:.2f}'
        cv2.putText(
            output,
            text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 210, 40),
            1,
            cv2.LINE_AA,
        )
    return output, (perf_counter() - start) * 1000.0


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = RemDetDetector(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        deploy=args.switch_to_deploy,
        precision=args.precision,
    )

    probe = cv2.VideoCapture(str(video_path))
    ok, warmup_frame = probe.read()
    fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    probe.release()
    if not ok:
        raise RuntimeError(f'Cannot read video: {video_path}')

    detector.warmup(warmup_frame, iterations=args.warmup)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    latency_samples = {
        'decode_ms': [],
        'preprocess_ms': [],
        'model_ms': [],
        'gpu_model_ms': [],
        'filter_ms': [],
        'pipeline_ms': [],
        'render_ms': [],
        'write_ms': [],
        'total_no_render_ms': [],
        'total_with_output_ms': [],
    }
    frame_records: list[dict] = []
    writer = None
    output_video = output_dir / 'visualization.mp4'

    if args.save_video:
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps if fps > 0 else 25.0,
            (width, height),
        )

    measured_frames = 0
    for repeat_index in range(max(1, args.repeats)):
        capture = cv2.VideoCapture(str(video_path))
        frame_index = 0
        while True:
            decode_start = perf_counter()
            ok, frame = capture.read()
            decode_ms = (perf_counter() - decode_start) * 1000.0
            if not ok:
                break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            detections, timings = detector.predict(
                frame, score_threshold=args.raw_score_thr)
            total_no_render_ms = decode_ms + timings['pipeline_ms']
            render_ms = 0.0
            write_ms = 0.0

            if writer is not None and repeat_index == 0:
                rendered, render_ms = draw_detections(
                    frame, detections, args.render_score_thr)
                write_start = perf_counter()
                writer.write(rendered)
                write_ms = (perf_counter() - write_start) * 1000.0

            latency_samples['decode_ms'].append(decode_ms)
            for key in (
                    'preprocess_ms', 'model_ms', 'gpu_model_ms',
                    'filter_ms', 'pipeline_ms'):
                latency_samples[key].append(timings[key])
            latency_samples['render_ms'].append(render_ms)
            latency_samples['write_ms'].append(write_ms)
            latency_samples['total_no_render_ms'].append(total_no_render_ms)
            latency_samples['total_with_output_ms'].append(
                total_no_render_ms + render_ms + write_ms)
            measured_frames += 1

            if repeat_index == 0:
                frame_records.append({
                    'frame_id': frame_index,
                    'timestamp_ms': (
                        frame_index / fps * 1000.0 if fps > 0 else None),
                    'detections': [detection.to_dict() for detection in detections],
                })
            frame_index += 1
        capture.release()

    if writer is not None:
        writer.release()

    latency = {
        name: summarize(values) for name, values in latency_samples.items()
    }
    mean_core_ms = latency['pipeline_ms']['mean']
    mean_total_ms = latency['total_no_render_ms']['mean']
    summary = {
        'experiment': args.experiment,
        'video': str(video_path),
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_size_mb': Path(args.checkpoint).stat().st_size / 1024**2,
        'device': args.device,
        'precision': args.precision,
        'input_size': args.input_size,
        'raw_score_threshold': args.raw_score_thr,
        'render_score_threshold': args.render_score_thr,
        'warmup_frames': args.warmup,
        'repeats': args.repeats,
        'source': {
            'fps': fps,
            'frame_count': source_frames,
            'width': width,
            'height': height,
        },
        'measured_frames': measured_frames,
        'parameter_count': detector.parameter_count,
        'parameter_count_m': detector.parameter_count / 1_000_000,
        'deploy_modules_converted': detector.deploy_modules_converted,
        'deployment_graph': args.switch_to_deploy,
        'fps_core': 1000.0 / mean_core_ms if mean_core_ms > 0 else 0.0,
        'fps_end_to_end_no_render': (
            1000.0 / mean_total_ms if mean_total_ms > 0 else 0.0),
        'gpu_peak_memory_mb': (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available() else 0.0),
        'gpu_peak_reserved_memory_mb': (
            torch.cuda.max_memory_reserved() / 1024**2
            if torch.cuda.is_available() else 0.0),
        'latency': latency,
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'mmcv': mmcv.__version__,
            'mmengine': mmengine.__version__,
            'gpu': (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else 'cpu'),
        },
    }

    with (output_dir / 'frames.jsonl').open('w', encoding='utf-8') as file:
        for record in frame_records:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')
    with (output_dir / 'latency_samples.json').open(
            'w', encoding='utf-8') as file:
        json.dump(latency_samples, file, ensure_ascii=False, indent=2)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
