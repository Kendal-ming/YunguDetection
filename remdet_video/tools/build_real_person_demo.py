"""Run RemDet-S on a real drone video and render a human-detection demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from remdet_video.core.detector import RemDetDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(
    r'C:\Users\xh\Desktop\新建文件夹\clip\9月1日\9月1日-1.mp4')
DEFAULT_OUTPUT = (
    PROJECT_ROOT / 'work_dirs/real_video_person_demo/demo'
    / 'remdet_real_drone_human_demo.mp4')
HUMAN_CLASSES = {'pedestrian', 'people'}
CLASS_COLORS = {
    'pedestrian': (55, 225, 90),
    'people': (0, 190, 255),
}


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = (
        max(0.0, second[2] - second[0])
        * max(0.0, second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_humans(
    detections: list[dict], iou_threshold: float = 0.40,
) -> list[dict]:
    """Apply presentation-only, class-agnostic NMS to the two human labels."""
    candidates = sorted(
        detections, key=lambda item: item['score'], reverse=True)
    kept: list[dict] = []
    for candidate in candidates:
        if all(bbox_iou(candidate['bbox'], item['bbox']) < iou_threshold
               for item in kept):
            kept.append(candidate)
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--config', type=Path,
        default=PROJECT_ROOT / 'config_remdet/remdet/remdet_s-300e_visdrone.py')
    parser.add_argument(
        '--checkpoint', type=Path,
        default=PROJECT_ROOT / 'checkpoints/remdet_s_weights_only.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threshold', type=float, default=0.30)
    parser.add_argument('--start-seconds', type=float, default=0.0)
    parser.add_argument('--duration-seconds', type=float, default=8.0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    return parser.parse_args()


def alpha_panel(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)


def put_text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 2,
) -> None:
    cv2.putText(
        image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image even when its Windows path contains non-ASCII text."""
    suffix = path.suffix or '.jpg'
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f'Cannot encode image as {suffix}')
    encoded.tofile(str(path))


def resize_box(
    bbox: list[float], source_width: int, source_height: int,
    output_width: int, output_height: int,
) -> tuple[int, int, int, int]:
    sx = output_width / source_width
    sy = output_height / source_height
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * sx)), int(round(y1 * sy)),
        int(round(x2 * sx)), int(round(y2 * sy)))


def draw_closeup(
    output: np.ndarray,
    source: np.ndarray,
    detection: dict,
    scaled_bbox: tuple[int, int, int, int],
) -> None:
    """Show a magnified source crop so tiny aerial targets are visible."""
    source_height, source_width = source.shape[:2]
    bx1, by1, bx2, by2 = detection['bbox']
    center_x = (bx1 + bx2) / 2.0
    center_y = (by1 + by2) / 2.0
    target_w = max(320, int((bx2 - bx1) * 4.0))
    target_h = max(240, int((by2 - by1) * 4.0))
    crop_w = min(source_width, target_w)
    crop_h = min(source_height, target_h)
    crop_x1 = int(np.clip(center_x - crop_w / 2, 0, source_width - crop_w))
    crop_y1 = int(np.clip(center_y - crop_h / 2, 0, source_height - crop_h))
    crop = source[crop_y1:crop_y1 + crop_h,
                  crop_x1:crop_x1 + crop_w].copy()

    inset_w, inset_h = 336, 252
    # Keep the inset on the opposite side of the main detection.
    box_center_x = (scaled_bbox[0] + scaled_bbox[2]) / 2
    inset_x = 28 if box_center_x > output.shape[1] / 2 else output.shape[1] - 364
    inset_y = 114
    crop = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_CUBIC)
    rel_x1 = int(round((bx1 - crop_x1) / crop_w * inset_w))
    rel_y1 = int(round((by1 - crop_y1) / crop_h * inset_h))
    rel_x2 = int(round((bx2 - crop_x1) / crop_w * inset_w))
    rel_y2 = int(round((by2 - crop_y1) / crop_h * inset_h))
    color = CLASS_COLORS[detection['class_name']]
    cv2.rectangle(crop, (rel_x1, rel_y1), (rel_x2, rel_y2), color, 3)

    alpha_panel(
        output, (inset_x - 6, inset_y - 34),
        (inset_x + inset_w + 6, inset_y + inset_h + 6), (8, 13, 18), 0.92)
    put_text(output, 'HIGHEST-CONFIDENCE CLOSE-UP',
             (inset_x, inset_y - 10), 0.48, (235, 235, 235), 1)
    output[inset_y:inset_y + inset_h,
           inset_x:inset_x + inset_w] = crop
    cv2.rectangle(
        output, (inset_x, inset_y),
        (inset_x + inset_w, inset_y + inset_h), color, 3)


def render_frame(
    source: np.ndarray,
    detections: list[dict],
    threshold: float,
    source_time: float,
    frame_number: int,
    model_ms: float,
    width: int,
    height: int,
    source_label: str | None = None,
) -> np.ndarray:
    source_height, source_width = source.shape[:2]
    output = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    human = [
        item for item in detections
        if item['class_name'] in HUMAN_CLASSES and item['score'] >= threshold
    ]
    human.sort(key=lambda item: item['score'], reverse=True)

    alpha_panel(output, (0, 0), (width, 92), (10, 16, 22), 0.88)
    alpha_panel(output, (0, height - 58), (width, height), (10, 16, 22), 0.88)
    put_text(output, 'RemDet-S  |  REAL DRONE FOOTAGE', (27, 36), 0.80)
    put_text(output, 'HUMAN DETECTION  (pedestrian + people)',
             (28, 72), 0.57, (215, 220, 225), 1)

    found = bool(human)
    status_color = (55, 225, 90) if found else (0, 190, 255)
    status = f'HUMAN DETECTED  x{len(human)}' if found else 'SCANNING'
    cv2.circle(output, (936, 47), 11, status_color, -1, cv2.LINE_AA)
    put_text(output, status, (960, 57), 0.72, status_color, 2)

    scaled_boxes: list[tuple[int, int, int, int]] = []
    for detection_index, detection in enumerate(human):
        x1, y1, x2, y2 = resize_box(
            detection['bbox'], source_width, source_height, width, height)
        scaled_boxes.append((x1, y1, x2, y2))
        color = CLASS_COLORS[detection['class_name']]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 4)
        label = f"HUMAN {detection_index + 1}  {detection['score']:.3f}"
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.49, 2)
        label_y1 = max(94, y1 - label_h - 12 - detection_index * 28)
        cv2.rectangle(
            output, (max(0, x1), label_y1),
            (min(width - 1, x1 + label_w + 12), label_y1 + label_h + 10),
            color, -1)
        put_text(output, label, (max(3, x1 + 6), label_y1 + label_h + 3),
                 0.49, (8, 16, 8), 2)

    if human:
        draw_closeup(output, source, human[0], scaled_boxes[0])

    put_text(
        output, f'SOURCE {source_time:05.2f}s   FRAME {frame_number:04d}',
        (27, height - 21), 0.52, (225, 225, 225), 1)
    if source_label:
        put_text(output, source_label, (535, height - 21),
                 0.49, (80, 225, 110), 1)
    put_text(
        output, f'CONF THR {threshold:.2f}   MODEL {model_ms:05.1f} ms   RTX 5080',
        (760, height - 21), 0.49, (205, 210, 215), 1)
    return output


def title_card(background: np.ndarray, width: int, height: int) -> np.ndarray:
    card = cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA)
    alpha_panel(card, (0, 0), (width, height), (5, 10, 16), 0.77)
    put_text(card, 'RemDet-S', (70, 215), 1.60, (65, 225, 100), 4)
    put_text(card, 'REAL DRONE VIDEO', (72, 286), 1.10, (250, 250, 250), 3)
    put_text(card, 'HUMAN DETECTION DEMO', (73, 350), 0.96, (0, 200, 255), 2)
    put_text(card, 'Target classes: pedestrian + people',
             (75, 412), 0.66, (220, 225, 230), 1)
    put_text(card, 'Every box shown is an actual RemDet model prediction.',
             (75, 633), 0.53, (180, 190, 200), 1)
    return card


def outro_card(background: np.ndarray, summary: dict) -> np.ndarray:
    card = background.copy()
    height, width = card.shape[:2]
    alpha_panel(card, (0, 0), (width, height), (5, 10, 16), 0.84)
    put_text(card, 'REAL-VIDEO TEST COMPLETE', (70, 244), 1.25,
             (65, 225, 100), 3)
    put_text(card, f"Human detections: {summary['detection_instances']}",
             (73, 322), 0.76)
    put_text(card, f"Frames triggered: {summary['triggered_frames']} / "
             f"{summary['processed_frames']}", (73, 370), 0.76)
    put_text(card, f"Maximum confidence: {summary['maximum_confidence']:.3f}",
             (73, 418), 0.76)
    put_text(card, 'Qualitative functional demo; the real footage has no labels.',
             (74, 630), 0.54, (185, 195, 205), 1)
    return card


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open input video: {input_path}')
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0:
        raise RuntimeError('Input video has no valid frame rate')

    start_frame = int(round(args.start_seconds * source_fps))
    requested_frames = int(round(args.duration_seconds * source_fps))
    end_frame = min(source_frames, start_frame + requested_frames)
    if start_frame >= end_frame:
        raise RuntimeError('Selected segment is outside the source video')
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first_frame = capture.read()
    if not ok:
        raise RuntimeError('Cannot decode the first selected frame')
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print('Loading RemDet-S...')
    detector = RemDetDetector(
        config=args.config, checkpoint=args.checkpoint, device=args.device,
        deploy=False, precision='fp32')
    print(f'Warmup: {args.warmup} iterations')
    detector.warmup(first_frame, iterations=args.warmup)

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), source_fps,
        (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create output video: {output_path}')

    title = title_card(first_frame, args.width, args.height)
    for _ in range(int(round(source_fps * 0.8))):
        writer.write(title)

    prediction_path = output_path.with_suffix('.jsonl')
    timings: list[float] = []
    max_confidence = 0.0
    detection_instances = 0
    triggered_frames = 0
    strongest_frame: np.ndarray | None = None
    strongest_score = -1.0
    last_rendered = title
    processed = 0

    with prediction_path.open('w', encoding='utf-8') as stream:
        for frame_index in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            detections, timing = detector.predict(frame, score_threshold=0.001)
            detection_dicts = [item.to_dict() for item in detections]
            raw_human = [
                item for item in detection_dicts
                if item['class_name'] in HUMAN_CLASSES
                and item['score'] >= args.threshold
            ]
            human = deduplicate_humans(raw_human)
            if human:
                triggered_frames += 1
                detection_instances += len(human)
                frame_max = human[0]['score']
                max_confidence = max(max_confidence, frame_max)
            else:
                frame_max = 0.0

            rendered = render_frame(
                frame, human, args.threshold, frame_index / source_fps,
                frame_index + 1, timing['model_ms'], args.width, args.height)
            writer.write(rendered)
            last_rendered = rendered
            if frame_max > strongest_score:
                strongest_score = frame_max
                strongest_frame = rendered.copy()

            stream.write(json.dumps({
                'source_frame': frame_index + 1,
                'source_time_seconds': frame_index / source_fps,
                'human_detections': human,
                'timings_ms': timing,
            }, ensure_ascii=False) + '\n')
            timings.append(timing['pipeline_ms'])
            processed += 1
            if processed % 30 == 0:
                print(f'Processed {processed}/{end_frame - start_frame} frames')

    capture.release()
    if processed == 0:
        writer.release()
        raise RuntimeError('No frames were processed')

    summary = {
        'passed': triggered_frames > 0,
        'input': str(input_path),
        'output': str(output_path),
        'predictions': str(prediction_path),
        'model': 'RemDet-S',
        'human_classes': sorted(HUMAN_CLASSES),
        'confidence_threshold': args.threshold,
        'source_video': {
            'width': source_width,
            'height': source_height,
            'fps': source_fps,
            'frame_count': source_frames,
        },
        'segment': {
            'start_seconds': args.start_seconds,
            'requested_duration_seconds': args.duration_seconds,
        },
        'processed_frames': processed,
        'triggered_frames': triggered_frames,
        'triggered_frame_ratio': triggered_frames / processed,
        'detection_instances': detection_instances,
        'maximum_confidence': max_confidence,
        'mean_pipeline_ms': mean(timings),
        'disclosure': (
            'Qualitative functional test. The real video has no ground-truth '
            'annotations, so this is not an accuracy or recall measurement.'),
    }

    outro = outro_card(last_rendered, summary)
    for _ in range(int(round(source_fps * 1.0))):
        writer.write(outro)
    writer.release()

    thumbnail_path = output_path.with_suffix('.jpg')
    if strongest_frame is None:
        strongest_frame = last_rendered
    write_image(thumbnail_path, strongest_frame)

    output_capture = cv2.VideoCapture(str(output_path))
    output_frames = int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = float(output_capture.get(cv2.CAP_PROP_FPS))
    output_width = int(output_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(output_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decodable, _ = output_capture.read()
    output_capture.release()
    summary['output_video'] = {
        'decodable': bool(decodable),
        'width': output_width,
        'height': output_height,
        'fps': output_fps,
        'frame_count': output_frames,
        'duration_seconds': output_frames / output_fps,
        'size_mb': output_path.stat().st_size / 1024**2,
    }
    summary['thumbnail'] = str(thumbnail_path)
    summary_path = output_path.with_suffix('.json')
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
