"""Cache low-threshold RemDet predictions for first-appearance events."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import mmcv
import mmengine
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remdet_video.core.detector import RemDetDetector  # noqa: E402


DEFAULT_EVENTS = (
    PROJECT_ROOT
    / 'work_dirs/first_appearance/manifest/target_candidate_events.json')
DEFAULT_OUTPUT = (
    PROJECT_ROOT / 'work_dirs/first_appearance/inference/remdet_s_fp32')
DEFAULT_CONFIG = (
    PROJECT_ROOT / 'config_remdet/remdet/remdet_s-300e_visdrone.py')
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / 'checkpoints/remdet_s_weights_only.pth')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, default=DEFAULT_EVENTS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--splits', nargs='+', default=['train', 'test-dev'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--precision', choices=('fp32', 'amp-fp16'), default='fp32')
    parser.add_argument('--raw-score-thr', type=float, default=0.001)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--max-followup-frames', type=int, default=30)
    parser.add_argument(
        '--single-label', action='store_true',
        help=('Keep only the highest-scoring class at each prediction '
              'location before NMS. The published config is multi-label.'))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def main() -> None:
    args = parse_args()
    events_path = args.events.resolve()
    output_dir = args.output_dir.resolve()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    for required in (events_path, config_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    events = json.loads(events_path.read_text(encoding='utf-8'))
    selected = [event for event in events if event['split'] in args.splits]
    if not selected:
        raise ValueError('No events remain after split filtering')

    # Several class events can share a sequence.  Process each source frame
    # once and retain only classes relevant to that sequence.
    by_sequence: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in selected:
        by_sequence[(event['split'], event['sequence'])].append(event)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / 'predictions.jsonl'
    metadata_path = output_dir / 'metadata.json'

    detector = RemDetDetector(
        config=config_path,
        checkpoint=checkpoint_path,
        device=args.device,
        precision=args.precision,
    )
    if args.single_label:
        detector.model.bbox_head.test_cfg.multi_label = False

    first_event = selected[0]
    first_frame_path = (
        Path(first_event['sequence_dir']) / '0000001.jpg')
    warmup_frame = cv2.imread(str(first_frame_path), cv2.IMREAD_COLOR)
    if warmup_frame is None:
        raise RuntimeError(f'Cannot read warmup image: {first_frame_path}')
    detector.warmup(warmup_frame, iterations=args.warmup)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = perf_counter()
    timing_samples: dict[str, list[float]] = defaultdict(list)
    total_frames = 0
    sequence_summaries: list[dict] = []

    with predictions_path.open('w', encoding='utf-8') as stream:
        for sequence_index, ((split, sequence), sequence_events) in enumerate(
                sorted(by_sequence.items()), start=1):
            sequence_dir = Path(sequence_events[0]['sequence_dir'])
            relevant_classes = sorted({
                event['class_name'] for event in sequence_events
            })
            stop_frame = min(
                int(sequence_events[0]['frame_count']),
                max(
                    max(
                        int(event['first_visible_frame']),
                        int(event['first_eligible_frame'] or 0),
                    ) + args.max_followup_frames
                    for event in sequence_events
                ),
            )
            sequence_started = perf_counter()
            kept_detections = 0
            for frame_id in range(1, stop_frame + 1):
                frame_path = sequence_dir / f'{frame_id:07d}.jpg'
                decode_started = perf_counter()
                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                decode_ms = (perf_counter() - decode_started) * 1000.0
                if frame is None:
                    raise RuntimeError(f'Cannot read image: {frame_path}')

                detections, timings = detector.predict(
                    frame, score_threshold=args.raw_score_thr)
                retained = [
                    detection.to_dict() for detection in detections
                    if detection.class_name in relevant_classes
                ]
                kept_detections += len(retained)
                record = {
                    'split': split,
                    'sequence': sequence,
                    'frame_id': frame_id,
                    'detections': retained,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                timing_samples['decode_ms'].append(decode_ms)
                for name, value in timings.items():
                    timing_samples[name].append(float(value))
                timing_samples['end_to_end_ms'].append(
                    decode_ms + float(timings['pipeline_ms']))
                total_frames += 1

            elapsed = perf_counter() - sequence_started
            sequence_summaries.append({
                'split': split,
                'sequence': sequence,
                'classes': relevant_classes,
                'frames_processed': stop_frame,
                'detections_retained': kept_detections,
                'elapsed_seconds': elapsed,
            })
            print(
                f'[{sequence_index}/{len(by_sequence)}] {split}/{sequence}: '
                f'{stop_frame} frames in {elapsed:.1f}s',
                flush=True,
            )

    elapsed_seconds = perf_counter() - started
    latency = {
        name: {
            'count': len(values),
            'mean_ms': mean(values) if values else None,
            'p50_ms': percentile(values, 50),
            'p95_ms': percentile(values, 95),
            'max_ms': max(values) if values else None,
        }
        for name, values in timing_samples.items()
    }
    metadata = {
        'passed': True,
        'events_file': str(events_path),
        'event_count': len(selected),
        'sequence_count': len(by_sequence),
        'splits': args.splits,
        'config': str(config_path),
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256(checkpoint_path),
        'device': args.device,
        'precision': args.precision,
        'raw_score_threshold': args.raw_score_thr,
        'warmup_iterations': args.warmup,
        'max_followup_frames': args.max_followup_frames,
        'multi_label': not args.single_label,
        'frames_processed': total_frames,
        'elapsed_seconds': elapsed_seconds,
        'effective_fps': total_frames / elapsed_seconds,
        'latency': latency,
        'gpu_peak_memory_mb': (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available() else 0.0),
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'mmcv': mmcv.__version__,
            'mmengine': mmengine.__version__,
            'gpu': (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else 'cpu'),
        },
        'sequences': sequence_summaries,
        'predictions': str(predictions_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
