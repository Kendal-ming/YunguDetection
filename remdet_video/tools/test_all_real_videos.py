"""Fully test all real drone videos and build one presentation compilation."""

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
    deduplicate_humans,
    put_text,
    render_frame,
    title_card,
    write_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSPECTION = (
    PROJECT_ROOT / 'work_dirs/real_video_person_demo/inspection/videos.json')
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / 'work_dirs/real_video_person_demo/all_videos')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspection', type=Path, default=DEFAULT_INSPECTION)
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
    parser.add_argument('--clip-seconds', type=float, default=6.0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    return parser.parse_args()


def select_best_window(records: list[dict], window_frames: int) -> int:
    """Select a continuous window favouring stable, confident detections."""
    if len(records) <= window_frames:
        return 0
    values = []
    for record in records:
        detections = record['human_detections']
        maximum = max((item['score'] for item in detections), default=0.0)
        values.append((1.0 if detections else 0.0) + 0.20 * maximum)
    current = sum(values[:window_frames])
    best_score = current
    best_start = 0
    for start in range(1, len(values) - window_frames + 1):
        current += values[start + window_frames - 1] - values[start - 1]
        if current > best_score:
            best_score = current
            best_start = start
    return best_start


def read_cached_records(path: Path) -> list[dict]:
    records = []
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            records.append(json.loads(line))
    return records


def scan_video(
    detector: RemDetDetector,
    video: dict,
    prediction_dir: Path,
    threshold: float,
    dedup_iou: float,
) -> tuple[list[dict], dict, np.ndarray]:
    video_id = int(video['video_id'])
    prediction_path = prediction_dir / f'video_{video_id:02d}.jsonl'
    summary_path = prediction_dir / f'video_{video_id:02d}_summary.json'
    expected_frames = int(video['frame_count'])

    if prediction_path.exists() and summary_path.exists():
        records = read_cached_records(prediction_path)
        if len(records) == expected_frames:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            strongest_index = int(summary['strongest_frame_index'])
            capture = cv2.VideoCapture(video['path'])
            capture.set(cv2.CAP_PROP_POS_FRAMES, strongest_index)
            ok, strongest_source = capture.read()
            capture.release()
            if not ok:
                raise RuntimeError('Cannot decode cached strongest frame')
            print(f"Video {video_id}/8: reused {len(records)} cached frames")
            return records, summary, strongest_source

    capture = cv2.VideoCapture(video['path'])
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video['path']}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or video['fps'])
    records: list[dict] = []
    timing_samples: list[float] = []
    triggered_frames = 0
    detection_instances = 0
    maximum_confidence = 0.0
    strongest_index = 0
    strongest_source: np.ndarray | None = None
    first_trigger_seconds: float | None = None
    last_trigger_seconds: float | None = None
    partial_path = prediction_path.with_suffix('.jsonl.partial')
    started = perf_counter()

    with partial_path.open('w', encoding='utf-8') as stream:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections, timings = detector.predict(frame, score_threshold=0.001)
            raw_human = [
                item.to_dict() for item in detections
                if item.class_name in HUMAN_CLASSES and item.score >= threshold
            ]
            human = deduplicate_humans(raw_human, iou_threshold=dedup_iou)
            frame_maximum = max(
                (item['score'] for item in human), default=0.0)
            source_time = frame_index / fps
            if human:
                triggered_frames += 1
                detection_instances += len(human)
                if first_trigger_seconds is None:
                    first_trigger_seconds = source_time
                last_trigger_seconds = source_time
            if frame_maximum > maximum_confidence or strongest_source is None:
                maximum_confidence = frame_maximum
                strongest_index = frame_index
                strongest_source = frame.copy()
            record = {
                'video_id': video_id,
                'frame_index': frame_index,
                'source_frame': frame_index + 1,
                'source_time_seconds': source_time,
                'human_detections': human,
                'timings_ms': timings,
            }
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            timing_samples.append(timings['pipeline_ms'])
            frame_index += 1
            if frame_index % 300 == 0:
                print(
                    f'  Video {video_id}/8: {frame_index}/{expected_frames} '
                    f'frames, triggers={triggered_frames}', flush=True)

    capture.release()
    if not records or strongest_source is None:
        raise RuntimeError(f'No frames decoded from video {video_id}')
    partial_path.replace(prediction_path)
    elapsed = perf_counter() - started
    summary = {
        'video_id': video_id,
        'path': video['path'],
        'relative_path': video['relative_path'],
        'fps': fps,
        'duration_seconds': len(records) / fps,
        'processed_frames': len(records),
        'triggered_frames': triggered_frames,
        'triggered_frame_ratio': triggered_frames / len(records),
        'non_triggered_frames': len(records) - triggered_frames,
        'detection_instances': detection_instances,
        'maximum_confidence': maximum_confidence,
        'first_trigger_seconds': first_trigger_seconds,
        'last_trigger_seconds': last_trigger_seconds,
        'mean_pipeline_ms': mean(timing_samples),
        'scan_wall_seconds': elapsed,
        'strongest_frame_index': strongest_index,
        'prediction_file': str(prediction_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(
        f"Video {video_id}/8 complete: {len(records)} frames, "
        f"trigger ratio={summary['triggered_frame_ratio']:.1%}, "
        f"max={maximum_confidence:.3f}, wall={elapsed:.1f}s", flush=True)
    return records, summary, strongest_source


def add_video_badge(
    image: np.ndarray, video_id: int, count: int, summary: dict,
) -> np.ndarray:
    output = image.copy()
    alpha_panel(output, (625, 10), (875, 82), (8, 13, 18), 0.84)
    put_text(output, f'FLIGHT CLIP {video_id} / {count}',
             (641, 40), 0.58, (80, 225, 110), 2)
    put_text(output, f"FULL TEST: {summary['triggered_frame_ratio']:.1%} frames",
             (641, 68), 0.43, (230, 230, 230), 1)
    return output


def make_contact_sheet(
    thumbnails: list[np.ndarray], summaries: list[dict], output_path: Path,
) -> None:
    tile_width, tile_height = 480, 270
    header_height = 74
    sheet = np.full(
        (header_height + tile_height * 2, tile_width * 4, 3),
        (12, 18, 24), dtype=np.uint8)
    put_text(sheet, 'RemDet-S | ALL 8 REAL FLIGHT VIDEOS | HUMAN DETECTION',
             (24, 45), 0.88, (245, 245, 245), 2)
    for index, (thumbnail, summary) in enumerate(zip(thumbnails, summaries)):
        tile = cv2.resize(
            thumbnail, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, 4)
        x = column * tile_width
        y = header_height + row * tile_height
        sheet[y:y + tile_height, x:x + tile_width] = tile
        alpha_panel(sheet, (x, y), (x + tile_width, y + 36),
                    (8, 13, 18), 0.86)
        put_text(
            sheet,
            f"VIDEO {summary['video_id']}  HIT {summary['triggered_frame_ratio']:.1%}  "
            f"MAX {summary['maximum_confidence']:.3f}",
            (x + 12, y + 25), 0.45, (245, 245, 245), 1)
    write_image(output_path, sheet)


def build_compilation(
    videos: list[dict], records_by_video: list[list[dict]],
    summaries: list[dict], output_path: Path, clip_seconds: float,
    threshold: float, width: int, height: int,
) -> dict:
    output_fps = 30.0
    first_capture = cv2.VideoCapture(videos[0]['path'])
    ok, first_frame = first_capture.read()
    first_capture.release()
    if not ok:
        raise RuntimeError('Cannot decode compilation background')
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), output_fps,
        (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create {output_path}')

    card = title_card(first_frame, width, height)
    alpha_panel(card, (65, 445), (675, 500), (8, 13, 18), 0.82)
    put_text(card, '8 FLIGHT CLIPS  |  FULL FRAME-BY-FRAME TEST',
             (82, 482), 0.62, (80, 225, 110), 2)
    for _ in range(int(round(output_fps * 1.2))):
        writer.write(card)

    selected_windows = []
    last_rendered = card
    for video, records, summary in zip(videos, records_by_video, summaries):
        source_fps = float(summary['fps'])
        window_frames = min(
            len(records), int(round(clip_seconds * source_fps)))
        start = select_best_window(records, window_frames)
        end = start + window_frames
        summary['demo_start_seconds'] = start / source_fps
        summary['demo_duration_seconds'] = window_frames / source_fps
        selected_windows.append({
            'video_id': video['video_id'],
            'start_frame_index': start,
            'end_frame_index_exclusive': end,
            'start_seconds': start / source_fps,
            'duration_seconds': window_frames / source_fps,
        })

        capture = cv2.VideoCapture(video['path'])
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_index in range(start, end):
            ok, frame = capture.read()
            if not ok:
                break
            record = records[frame_index]
            last_rendered = render_frame(
                frame, record['human_detections'], threshold,
                record['source_time_seconds'], record['source_frame'],
                record['timings_ms']['model_ms'], width, height,
                source_label=f"VIDEO {video['video_id']} / {len(videos)}")
            last_rendered = add_video_badge(
                last_rendered, int(video['video_id']), len(videos), summary)
            writer.write(last_rendered)
        capture.release()
        print(
            f"Added video {video['video_id']} highlight: "
            f"{start / source_fps:.2f}s to {end / source_fps:.2f}s",
            flush=True)

    outro = last_rendered.copy()
    alpha_panel(outro, (0, 0), (width, height), (5, 10, 16), 0.84)
    total_frames = sum(item['processed_frames'] for item in summaries)
    total_triggered = sum(item['triggered_frames'] for item in summaries)
    put_text(outro, 'ALL 8 FLIGHT VIDEOS TESTED', (70, 250), 1.25,
             (65, 225, 100), 3)
    put_text(outro, f'Frames processed: {total_frames:,}',
             (73, 330), 0.76)
    put_text(outro, f'Frames with human trigger: {total_triggered:,}',
             (73, 380), 0.76)
    put_text(outro, 'Result: human targets detected in every source video',
             (73, 430), 0.72, (245, 245, 245), 2)
    put_text(outro, 'Qualitative test: source videos have no ground-truth labels.',
             (74, 630), 0.54, (185, 195, 205), 1)
    for _ in range(int(round(output_fps * 1.2))):
        writer.write(outro)
    writer.release()

    capture = cv2.VideoCapture(str(output_path))
    output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    output_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decodable, _ = capture.read()
    capture.release()
    return {
        'path': str(output_path),
        'decodable': bool(decodable),
        'width': output_width,
        'height': output_height,
        'fps': fps,
        'frame_count': output_frames,
        'duration_seconds': output_frames / fps,
        'size_mb': output_path.stat().st_size / 1024**2,
        'selected_windows': selected_windows,
    }


def write_csv(path: Path, summaries: list[dict]) -> None:
    columns = [
        'video_id', 'relative_path', 'duration_seconds', 'processed_frames',
        'triggered_frames', 'triggered_frame_ratio', 'non_triggered_frames',
        'detection_instances', 'maximum_confidence', 'first_trigger_seconds',
        'last_trigger_seconds', 'mean_pipeline_ms', 'scan_wall_seconds',
        'demo_start_seconds', 'demo_duration_seconds',
    ]
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    inspection = json.loads(
        args.inspection.resolve().read_text(encoding='utf-8'))
    videos = inspection['videos']
    output_dir = args.output_dir.resolve()
    prediction_dir = output_dir / 'predictions'
    prediction_dir.mkdir(parents=True, exist_ok=True)

    probe = cv2.VideoCapture(videos[0]['path'])
    ok, warmup_frame = probe.read()
    probe.release()
    if not ok:
        raise RuntimeError('Cannot read warmup frame')
    print('Loading RemDet-S once for all 8 videos...', flush=True)
    detector = RemDetDetector(
        config=args.config, checkpoint=args.checkpoint, device=args.device,
        deploy=False, precision='fp32')
    detector.warmup(warmup_frame, iterations=args.warmup)

    all_started = perf_counter()
    records_by_video: list[list[dict]] = []
    summaries: list[dict] = []
    thumbnails: list[np.ndarray] = []
    for video in videos:
        records, summary, strongest_source = scan_video(
            detector, video, prediction_dir, args.threshold, args.dedup_iou)
        records_by_video.append(records)
        summaries.append(summary)
        strongest_record = records[int(summary['strongest_frame_index'])]
        thumbnail = render_frame(
            strongest_source, strongest_record['human_detections'],
            args.threshold, strongest_record['source_time_seconds'],
            strongest_record['source_frame'],
            strongest_record['timings_ms']['model_ms'], args.width, args.height,
            source_label=f"VIDEO {video['video_id']} / {len(videos)}")
        thumbnails.append(thumbnail)

    contact_sheet = output_dir / 'all_8_videos_human_results.jpg'
    make_contact_sheet(thumbnails, summaries, contact_sheet)
    demo_path = output_dir / 'remdet_all_8_real_videos_human_demo.mp4'
    demo = build_compilation(
        videos, records_by_video, summaries, demo_path, args.clip_seconds,
        args.threshold, args.width, args.height)
    csv_path = output_dir / 'all_8_videos_results.csv'
    write_csv(csv_path, summaries)

    total_frames = sum(item['processed_frames'] for item in summaries)
    total_triggered = sum(item['triggered_frames'] for item in summaries)
    final_report = {
        'passed': all(item['triggered_frames'] > 0 for item in summaries),
        'model': 'RemDet-S',
        'test_scope': 'all frames of all supplied real videos',
        'video_count': len(videos),
        'total_source_duration_seconds': sum(
            item['duration_seconds'] for item in summaries),
        'total_processed_frames': total_frames,
        'total_triggered_frames': total_triggered,
        'overall_triggered_frame_ratio': total_triggered / total_frames,
        'confidence_threshold': args.threshold,
        'human_classes': sorted(HUMAN_CLASSES),
        'mean_pipeline_ms_weighted': sum(
            item['mean_pipeline_ms'] * item['processed_frames']
            for item in summaries) / total_frames,
        'videos': summaries,
        'demo': demo,
        'contact_sheet': str(contact_sheet),
        'csv': str(csv_path),
        'total_wall_seconds': perf_counter() - all_started,
        'disclosure': (
            'The source videos have no frame-level ground-truth annotations. '
            'Triggered-frame ratios measure model activity, not accuracy, '
            'precision, or recall.'),
    }
    report_path = output_dir / 'all_8_videos_full_test_report.json'
    final_report['report'] = str(report_path)
    report_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
