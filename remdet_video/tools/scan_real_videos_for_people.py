"""Run a sparse RemDet scan over real videos for human-related classes."""

from __future__ import annotations

import argparse
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


DEFAULT_INSPECTION = (
    PROJECT_ROOT / 'work_dirs/real_video_person_demo/inspection/videos.json')
DEFAULT_OUTPUT = PROJECT_ROOT / 'work_dirs/real_video_person_demo/sparse_scan'
DEFAULT_CONFIG = (
    PROJECT_ROOT / 'config_remdet/remdet/remdet_s-300e_visdrone.py')
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'checkpoints/remdet_s_weights_only.pth'
TARGET_CLASSES = ('pedestrian', 'people')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspection', type=Path, default=DEFAULT_INSPECTION)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stride', type=int, default=30)
    parser.add_argument('--raw-score-thr', type=float, default=0.001)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--top-per-video', type=int, default=4)
    return parser.parse_args()


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or '.jpg', image)
    if not ok:
        raise RuntimeError(f'Cannot encode image: {path}')
    encoded.tofile(str(path))


def select_spaced(records: list[dict], count: int, spacing: int) -> list[dict]:
    selected = []
    for record in sorted(records, key=lambda item: item['max_score'], reverse=True):
        if all(
                record['video_id'] != item['video_id']
                or abs(record['frame_id'] - item['frame_id']) >= spacing
                for item in selected):
            selected.append(record)
            if len(selected) >= count:
                break
    return selected


def render_candidate(
    frame: np.ndarray,
    record: dict,
    width: int = 480,
    height: int = 270,
) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    output = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    sx, sy = width / source_width, height / source_height
    for detection in record['detections']:
        x1, y1, x2, y2 = [
            int(round(value)) for value in (
                detection['bbox'][0] * sx,
                detection['bbox'][1] * sy,
                detection['bbox'][2] * sx,
                detection['bbox'][3] * sy,
            )
        ]
        color = (
            (60, 230, 80)
            if detection['class_name'] == 'pedestrian'
            else (0, 195, 255))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            output,
            f'{detection["class_name"]} {detection["score"]:.3f}',
            (max(2, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, 31), (8, 12, 16), -1)
    cv2.addWeighted(overlay, 0.82, output, 0.18, 0.0, output)
    cv2.putText(
        output,
        f'VIDEO {record["video_id"]} | {record["time_seconds"]:.1f}s | '
        f'frame {record["frame_id"]}',
        (9, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return output


def read_frame(path: Path, frame_id: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f'Cannot read {path} frame {frame_id}')
    return frame


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError('--stride must be positive')
    inspection = json.loads(
        args.inspection.resolve().read_text(encoding='utf-8'))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    first_video = Path(inspection['videos'][0]['path'])
    probe = cv2.VideoCapture(str(first_video))
    ok, warmup_frame = probe.read()
    probe.release()
    if not ok:
        raise RuntimeError(f'Cannot read warmup frame from {first_video}')

    detector = RemDetDetector(
        config=args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        precision='fp32',
    )
    detector.warmup(warmup_frame, iterations=args.warmup)

    records = []
    timing_samples: dict[str, list[float]] = defaultdict(list)
    video_summaries = []
    started = perf_counter()
    for item in inspection['videos']:
        path = Path(item['path'])
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open {path}')
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_index = 0
        sampled = 0
        video_records = []
        video_started = perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.stride != 0:
                frame_index += 1
                continue
            detections, timings = detector.predict(
                frame, score_threshold=args.raw_score_thr)
            human = [
                detection.to_dict() for detection in detections
                if detection.class_name in TARGET_CLASSES
            ]
            record = {
                'video_id': int(item['video_id']),
                'video_path': str(path),
                'frame_id': frame_index + 1,
                'time_seconds': frame_index / fps,
                'detections': human,
                'max_score': max(
                    (detection['score'] for detection in human), default=0.0),
            }
            records.append(record)
            video_records.append(record)
            sampled += 1
            for name, value in timings.items():
                timing_samples[name].append(float(value))
            frame_index += 1
        capture.release()
        scores = [
            detection['score']
            for record in video_records
            for detection in record['detections']
        ]
        summary = {
            'video_id': int(item['video_id']),
            'path': str(path),
            'sampled_frames': sampled,
            'maximum_human_score': max(scores, default=0.0),
            'frames_with_score_ge_0_05': sum(
                record['max_score'] >= 0.05 for record in video_records),
            'frames_with_score_ge_0_10': sum(
                record['max_score'] >= 0.10 for record in video_records),
            'frames_with_score_ge_0_20': sum(
                record['max_score'] >= 0.20 for record in video_records),
            'elapsed_seconds': perf_counter() - video_started,
        }
        video_summaries.append(summary)
        print(
            f'VIDEO {item["video_id"]}: sampled={sampled}, '
            f'max_human_score={summary["maximum_human_score"]:.4f}, '
            f'>=0.10={summary["frames_with_score_ge_0_10"]}',
            flush=True,
        )

    predictions_path = output_dir / 'human_sparse_predictions.jsonl'
    with predictions_path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    candidates = []
    for item in inspection['videos']:
        video_records = [
            record for record in records
            if record['video_id'] == int(item['video_id'])
        ]
        candidates.extend(select_spaced(
            video_records, args.top_per_video, spacing=args.stride * 2))
    candidates.sort(key=lambda item: (item['video_id'], -item['max_score']))

    cell_width, cell_height = 480, 270
    columns = 4
    rows = int(np.ceil(len(candidates) / columns))
    contact_sheet = np.zeros(
        (rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for index, candidate in enumerate(candidates):
        frame = read_frame(
            Path(candidate['video_path']), int(candidate['frame_id']))
        rendered = render_candidate(frame, candidate, cell_width, cell_height)
        y = index // columns * cell_height
        x = index % columns * cell_width
        contact_sheet[y:y + cell_height, x:x + cell_width] = rendered
    contact_sheet_path = output_dir / 'top_human_candidates.jpg'
    write_image(contact_sheet_path, contact_sheet)

    elapsed = perf_counter() - started
    metadata = {
        'passed': True,
        'inspection': str(args.inspection.resolve()),
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'device': args.device,
        'target_classes': list(TARGET_CLASSES),
        'stride_frames': args.stride,
        'sample_interval_seconds_at_30fps': args.stride / 30.0,
        'raw_score_threshold': args.raw_score_thr,
        'sampled_frames': len(records),
        'elapsed_seconds': elapsed,
        'inference_pipeline_mean_ms': mean(timing_samples['pipeline_ms']),
        'video_summaries': video_summaries,
        'top_candidates': candidates,
        'artifacts': {
            'predictions': str(predictions_path),
            'contact_sheet': str(contact_sheet_path),
        },
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'mmcv': mmcv.__version__,
            'mmengine': mmengine.__version__,
            'gpu': torch.cuda.get_device_name(0),
        },
    }
    report_path = output_dir / 'summary.json'
    report_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
