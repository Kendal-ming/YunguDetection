"""Render all DJI_0042 frames in order using cached RemDet predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from remdet_video.tools.build_real_person_demo import (
    CLASS_COLORS,
    alpha_panel,
    put_text,
    resize_box,
    write_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(r'C:\Users\xh\Desktop\新建文件夹\DJI_0042.MP4')
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / 'work_dirs/real_video_person_demo/dji_0042'
    / 'predictions/video_01.jsonl')
DEFAULT_OUTPUT = (
    PROJECT_ROOT / 'work_dirs/real_video_person_demo/dji_0042'
    / 'remdet_dji_0042_full_continuous_detected.mp4')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--predictions', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--threshold', type=float, default=0.30)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    return parser.parse_args()


def render_clean_frame(
    source: np.ndarray,
    record: dict,
    threshold: float,
    total_frames: int,
    width: int,
    height: int,
) -> np.ndarray:
    source_height, source_width = source.shape[:2]
    output = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    detections = record['human_detections']

    alpha_panel(output, (0, 0), (width, 74), (10, 16, 22), 0.82)
    alpha_panel(output, (0, height - 48), (width, height), (10, 16, 22), 0.82)
    put_text(output, 'RemDet-S  |  FULL CONTINUOUS VIDEO',
             (26, 32), 0.75, (245, 245, 245), 2)
    put_text(output, 'HUMAN DETECTION  (pedestrian + people)',
             (27, 60), 0.49, (215, 220, 225), 1)

    found = bool(detections)
    status_color = (55, 225, 90) if found else (0, 190, 255)
    status = f'HUMAN DETECTED  x{len(detections)}' if found else 'SCANNING'
    cv2.circle(output, (958, 38), 10, status_color, -1, cv2.LINE_AA)
    put_text(output, status, (980, 48), 0.64, status_color, 2)

    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = resize_box(
            detection['bbox'], source_width, source_height, width, height)
        color = CLASS_COLORS.get(detection['class_name'], (55, 225, 90))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 4)
        label = f"HUMAN {index + 1}  {detection['score']:.3f}"
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 2)
        label_y = max(76, y1 - label_height - 10 - index * 25)
        label_x = max(0, min(x1, width - label_width - 12))
        cv2.rectangle(
            output, (label_x, label_y),
            (label_x + label_width + 12, label_y + label_height + 9),
            color, -1)
        put_text(output, label, (label_x + 6, label_y + label_height + 2),
                 0.47, (8, 16, 8), 2)

    frame_number = int(record['source_frame'])
    source_time = float(record['source_time_seconds'])
    progress = frame_number / total_frames
    put_text(output, f'DJI_0042   TIME {source_time:06.2f}s',
             (26, height - 17), 0.50, (230, 230, 230), 1)
    put_text(output, f'FRAME {frame_number:04d} / {total_frames:04d}',
             (515, height - 17), 0.50, (230, 230, 230), 1)
    put_text(output, f'CONF THR {threshold:.2f}   PROGRESS {progress:6.1%}',
             (913, height - 17), 0.47, (210, 215, 220), 1)
    return output


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    prediction_path = args.predictions.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open input video: {input_path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps,
        (args.width, args.height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f'Cannot create output video: {output_path}')

    processed = 0
    strongest_score = -1.0
    strongest_rendered: np.ndarray | None = None
    started = perf_counter()
    with prediction_path.open('r', encoding='utf-8') as predictions:
        for line in predictions:
            record = json.loads(line)
            ok, frame = capture.read()
            if not ok:
                break
            if int(record['frame_index']) != processed:
                raise RuntimeError(
                    f'Prediction/frame mismatch at index {processed}')
            rendered = render_clean_frame(
                frame, record, args.threshold, expected_frames,
                args.width, args.height)
            writer.write(rendered)
            frame_maximum = max(
                (item['score'] for item in record['human_detections']),
                default=0.0)
            if frame_maximum > strongest_score:
                strongest_score = frame_maximum
                strongest_rendered = rendered.copy()
            processed += 1
            if processed % 300 == 0:
                elapsed = perf_counter() - started
                print(
                    f'Rendered {processed}/{expected_frames} frames '
                    f'({processed / expected_frames:.1%}), '
                    f'{processed / elapsed:.1f} output FPS', flush=True)

    capture.release()
    writer.release()
    if processed != expected_frames:
        raise RuntimeError(
            f'Incomplete output: rendered {processed}, expected {expected_frames}')

    check = cv2.VideoCapture(str(output_path))
    output_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = float(check.get(cv2.CAP_PROP_FPS))
    output_width = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decodable, _ = check.read()
    check.release()
    checks = {
        'decodable': bool(decodable),
        'all_source_frames_rendered': processed == expected_frames,
        'frame_count_matches_source': output_frames == expected_frames,
        'fps_matches_source': abs(output_fps - fps) < 0.01,
        'duration_matches_source': abs(
            output_frames / output_fps - expected_frames / fps) < 0.05,
    }
    if not all(checks.values()):
        raise RuntimeError(f'Output validation failed: {checks}')

    thumbnail_path = output_path.with_suffix('.jpg')
    if strongest_rendered is not None:
        write_image(thumbnail_path, strongest_rendered)
    report = {
        'passed': True,
        'input': str(input_path),
        'predictions': str(prediction_path),
        'output': str(output_path),
        'rendering': 'continuous, no temporal cuts, no skipped frames',
        'checks': checks,
        'source': {
            'width': source_width,
            'height': source_height,
            'fps': fps,
            'frame_count': expected_frames,
            'duration_seconds': expected_frames / fps,
        },
        'output_video': {
            'width': output_width,
            'height': output_height,
            'fps': output_fps,
            'frame_count': output_frames,
            'duration_seconds': output_frames / output_fps,
            'size_mb': output_path.stat().st_size / 1024**2,
        },
        'render_wall_seconds': perf_counter() - started,
        'thumbnail': str(thumbnail_path),
    }
    report_path = output_path.with_suffix('.json')
    report['report'] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
