"""Fully test DJI_0042 and build a time-diverse human-detection demo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from remdet_video.core.detector import RemDetDetector
from remdet_video.tools.build_real_person_demo import (
    HUMAN_CLASSES,
    alpha_panel,
    put_text,
    render_frame,
    title_card,
    write_image,
)
from remdet_video.tools.test_all_real_videos import scan_video, select_best_window


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(r'C:\Users\xh\Desktop\新建文件夹\DJI_0042.MP4')
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / 'work_dirs/real_video_person_demo/dji_0042')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--config', type=Path,
        default=PROJECT_ROOT / 'config_remdet/remdet/remdet_s-300e_visdrone.py')
    parser.add_argument(
        '--checkpoint', type=Path,
        default=PROJECT_ROOT / 'checkpoints/remdet_s_weights_only.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threshold', type=float, default=0.30)
    parser.add_argument('--dedup-iou', type=float, default=0.40)
    parser.add_argument('--time-buckets', type=int, default=8)
    parser.add_argument('--highlight-seconds', type=float, default=5.0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    return parser.parse_args()


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open input video: {path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    result = {
        'video_id': 1,
        'path': str(path),
        'relative_path': path.name,
        'size_mb': path.stat().st_size / 1024**2,
        'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': fps,
        'frame_count': frame_count,
        'duration_seconds': frame_count / fps,
    }
    capture.release()
    return result


def select_time_diverse_windows(
    records: list[dict], fps: float, bucket_count: int,
    highlight_seconds: float,
) -> list[dict]:
    window_frames = max(1, int(round(highlight_seconds * fps)))
    total = len(records)
    windows = []
    for bucket_index in range(bucket_count):
        bucket_start = int(round(bucket_index * total / bucket_count))
        bucket_end = int(round((bucket_index + 1) * total / bucket_count))
        bucket = records[bucket_start:bucket_end]
        if not bucket:
            continue
        local_window = min(window_frames, len(bucket))
        local_start = select_best_window(bucket, local_window)
        start = bucket_start + local_start
        end = start + local_window
        selected = records[start:end]
        triggered = sum(bool(item['human_detections']) for item in selected)
        maximum = max(
            (detection['score'] for item in selected
             for detection in item['human_detections']), default=0.0)
        # Do not fill a presentation demo with a section where the model never
        # reported a person. The full scan still includes every such frame.
        if triggered == 0:
            continue
        windows.append({
            'highlight_index': len(windows) + 1,
            'bucket_index': bucket_index,
            'start_frame_index': start,
            'end_frame_index_exclusive': end,
            'start_seconds': start / fps,
            'duration_seconds': local_window / fps,
            'triggered_frames': triggered,
            'triggered_frame_ratio': triggered / local_window,
            'maximum_confidence': maximum,
        })
    return windows


def aggregate_timeline(
    records: list[dict], fps: float, path: Path,
) -> list[dict]:
    duration_seconds = int(np.ceil(len(records) / fps))
    rows = []
    for second in range(duration_seconds):
        start = int(round(second * fps))
        end = min(len(records), int(round((second + 1) * fps)))
        segment = records[start:end]
        triggered = sum(bool(item['human_detections']) for item in segment)
        maximum = max(
            (detection['score'] for item in segment
             for detection in item['human_detections']), default=0.0)
        rows.append({
            'second': second,
            'frames': len(segment),
            'triggered_frames': triggered,
            'triggered_frame_ratio': triggered / len(segment) if segment else 0,
            'maximum_confidence': maximum,
        })
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def add_highlight_badge(
    image: np.ndarray, index: int, count: int, window: dict,
) -> np.ndarray:
    output = image.copy()
    alpha_panel(output, (625, 10), (875, 82), (8, 13, 18), 0.84)
    put_text(output, f'HIGHLIGHT {index} / {count}',
             (645, 40), 0.58, (80, 225, 110), 2)
    put_text(output, f"WINDOW HIT: {window['triggered_frame_ratio']:.1%}",
             (645, 68), 0.43, (230, 230, 230), 1)
    return output


def make_highlight_sheet(
    path: Path, frames: list[np.ndarray], windows: list[dict],
) -> None:
    columns = 4
    rows = int(np.ceil(len(frames) / columns))
    tile_width, tile_height = 480, 270
    header_height = 74
    sheet = np.full(
        (header_height + rows * tile_height, columns * tile_width, 3),
        (12, 18, 24), dtype=np.uint8)
    put_text(sheet, 'RemDet-S | DJI_0042 | TIME-DIVERSE HUMAN HIGHLIGHTS',
             (24, 45), 0.88, (245, 245, 245), 2)
    for index, (frame, window) in enumerate(zip(frames, windows)):
        tile = cv2.resize(
            frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x, y = column * tile_width, header_height + row * tile_height
        sheet[y:y + tile_height, x:x + tile_width] = tile
        alpha_panel(sheet, (x, y), (x + tile_width, y + 36),
                    (8, 13, 18), 0.86)
        put_text(
            sheet,
            f"H{index + 1}  TIME {window['start_seconds']:.1f}s  "
            f"HIT {window['triggered_frame_ratio']:.0%}  "
            f"MAX {window['maximum_confidence']:.3f}",
            (x + 12, y + 25), 0.43, (245, 245, 245), 1)
    write_image(path, sheet)


def build_demo(
    video: dict, records: list[dict], windows: list[dict], output: Path,
    threshold: float, width: int, height: int,
) -> tuple[dict, list[np.ndarray]]:
    fps = float(video['fps'])
    probe = cv2.VideoCapture(video['path'])
    ok, background = probe.read()
    probe.release()
    if not ok:
        raise RuntimeError('Cannot decode demo background')
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create demo: {output}')

    card = title_card(background, width, height)
    alpha_panel(card, (65, 445), (760, 500), (8, 13, 18), 0.82)
    put_text(card, 'DJI_0042  |  FULL 5-MINUTE FRAME-BY-FRAME TEST',
             (82, 482), 0.59, (80, 225, 110), 2)
    for _ in range(int(round(fps * 1.2))):
        writer.write(card)

    highlight_frames: list[np.ndarray] = []
    last_rendered = card
    for index, window in enumerate(windows, start=1):
        capture = cv2.VideoCapture(video['path'])
        start = int(window['start_frame_index'])
        end = int(window['end_frame_index_exclusive'])
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        best_score = -1.0
        best_rendered: np.ndarray | None = None
        for frame_index in range(start, end):
            ok, frame = capture.read()
            if not ok:
                break
            record = records[frame_index]
            last_rendered = render_frame(
                frame, record['human_detections'], threshold,
                record['source_time_seconds'], record['source_frame'],
                record['timings_ms']['model_ms'], width, height,
                source_label='DJI_0042')
            last_rendered = add_highlight_badge(
                last_rendered, index, len(windows), window)
            writer.write(last_rendered)
            score = max(
                (item['score'] for item in record['human_detections']),
                default=0.0)
            if score > best_score:
                best_score = score
                best_rendered = last_rendered.copy()
        capture.release()
        if best_rendered is not None:
            highlight_frames.append(best_rendered)
        print(
            f"Added highlight {index}/{len(windows)}: "
            f"{window['start_seconds']:.2f}s to "
            f"{window['start_seconds'] + window['duration_seconds']:.2f}s",
            flush=True)

    outro = last_rendered.copy()
    alpha_panel(outro, (0, 0), (width, height), (5, 10, 16), 0.84)
    total_triggered = sum(bool(item['human_detections']) for item in records)
    maximum = max(
        (detection['score'] for item in records
         for detection in item['human_detections']), default=0.0)
    put_text(outro, 'DJI_0042 TEST COMPLETE', (70, 250), 1.30,
             (65, 225, 100), 3)
    put_text(outro, f"Frames processed: {len(records):,}",
             (73, 330), 0.76)
    put_text(outro, f"Frames with human trigger: {total_triggered:,}",
             (73, 380), 0.76)
    put_text(outro, f'Maximum confidence: {maximum:.3f}',
             (73, 430), 0.76)
    put_text(outro, 'Selected highlights from the complete source-video test.',
             (74, 630), 0.54, (185, 195, 205), 1)
    for _ in range(int(round(fps * 1.2))):
        writer.write(outro)
    writer.release()

    check = cv2.VideoCapture(str(output))
    frame_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = float(check.get(cv2.CAP_PROP_FPS))
    output_width = int(check.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decodable, _ = check.read()
    check.release()
    return ({
        'path': str(output),
        'decodable': bool(decodable),
        'width': output_width,
        'height': output_height,
        'fps': output_fps,
        'frame_count': frame_count,
        'duration_seconds': frame_count / output_fps,
        'size_mb': output.stat().st_size / 1024**2,
    }, highlight_frames)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    prediction_dir = output_dir / 'predictions'
    prediction_dir.mkdir(parents=True, exist_ok=True)
    video = inspect_video(input_path)
    print(json.dumps(video, ensure_ascii=True, indent=2), flush=True)

    probe = cv2.VideoCapture(str(input_path))
    ok, warmup_frame = probe.read()
    probe.release()
    if not ok:
        raise RuntimeError('Cannot decode warmup frame')
    print('Loading RemDet-S...', flush=True)
    detector = RemDetDetector(
        config=args.config, checkpoint=args.checkpoint, device=args.device,
        deploy=False, precision='fp32')
    detector.warmup(warmup_frame, iterations=args.warmup)

    started = perf_counter()
    records, scan_summary, strongest_source = scan_video(
        detector, video, prediction_dir, args.threshold, args.dedup_iou)
    windows = select_time_diverse_windows(
        records, video['fps'], args.time_buckets, args.highlight_seconds)
    if not windows:
        # Preserve a useful output even if no person clears the threshold.
        window_frames = min(
            len(records), int(round(args.highlight_seconds * video['fps'])))
        windows = [{
            'highlight_index': 1,
            'bucket_index': 0,
            'start_frame_index': 0,
            'end_frame_index_exclusive': window_frames,
            'start_seconds': 0.0,
            'duration_seconds': window_frames / video['fps'],
            'triggered_frames': 0,
            'triggered_frame_ratio': 0.0,
            'maximum_confidence': 0.0,
        }]

    timeline_path = output_dir / 'dji_0042_timeline_per_second.csv'
    timeline = aggregate_timeline(records, video['fps'], timeline_path)
    demo_path = output_dir / 'remdet_dji_0042_human_demo.mp4'
    demo, highlight_frames = build_demo(
        video, records, windows, demo_path, args.threshold,
        args.width, args.height)
    sheet_path = output_dir / 'dji_0042_human_highlights.jpg'
    make_highlight_sheet(sheet_path, highlight_frames, windows)

    strongest_record = records[int(scan_summary['strongest_frame_index'])]
    strongest_rendered = render_frame(
        strongest_source, strongest_record['human_detections'], args.threshold,
        strongest_record['source_time_seconds'], strongest_record['source_frame'],
        strongest_record['timings_ms']['model_ms'], args.width, args.height,
        source_label='DJI_0042')
    strongest_path = output_dir / 'dji_0042_strongest_detection.jpg'
    write_image(strongest_path, strongest_rendered)

    triggered = int(scan_summary['triggered_frames'])
    report = {
        'passed': triggered > 0,
        'model': 'RemDet-S',
        'input': video,
        'test_scope': 'every frame of the complete source video',
        'human_classes': sorted(HUMAN_CLASSES),
        'confidence_threshold': args.threshold,
        'processed_frames': len(records),
        'triggered_frames': triggered,
        'triggered_frame_ratio': triggered / len(records),
        'maximum_confidence': scan_summary['maximum_confidence'],
        'first_trigger_seconds': scan_summary['first_trigger_seconds'],
        'last_trigger_seconds': scan_summary['last_trigger_seconds'],
        'mean_pipeline_ms': scan_summary['mean_pipeline_ms'],
        'scan_wall_seconds': scan_summary['scan_wall_seconds'],
        'per_second_records': len(timeline),
        'highlight_windows': windows,
        'demo': demo,
        'strongest_detection_image': str(strongest_path),
        'highlight_contact_sheet': str(sheet_path),
        'timeline_csv': str(timeline_path),
        'predictions': scan_summary['prediction_file'],
        'total_wall_seconds_this_run': perf_counter() - started,
        'disclosure': (
            'The source video has no frame-level ground-truth annotations. '
            'Triggered-frame ratio measures model activity, not accuracy, '
            'precision, or recall. The demo contains automatically selected '
            'time-diverse highlights from the full-video test.'),
    }
    report_path = output_dir / 'dji_0042_full_test_report.json'
    report['report'] = str(report_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
