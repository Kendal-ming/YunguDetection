"""Render a short presentation demo from a validated first-appearance event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE = Path(
    r'C:\Users\xh\Desktop\WindyLab\datasets\VisDrone2019-VID-test-dev'
    r'\sequences\uav0000370_00001_v')
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / 'work_dirs/first_appearance/inference/remdet_s_fp32/predictions.jsonl')
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / 'work_dirs/first_appearance/demo/remdet_van_first_appearance_demo.mp4')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-dir', type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument('--predictions', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--sequence', default='uav0000370_00001_v')
    parser.add_argument('--target-class', default='van')
    parser.add_argument('--threshold', type=float, default=0.36)
    parser.add_argument('--first-visible-frame', type=int, default=43)
    parser.add_argument('--output-fps', type=float, default=30.0)
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


def text(
    image: np.ndarray,
    value: str,
    position: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def scaled_box(
    bbox: list[float],
    source_shape: tuple[int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    source_height, source_width = source_shape[:2]
    sx = width / source_width
    sy = height / source_height
    return tuple(int(round(value)) for value in (
        bbox[0] * sx,
        bbox[1] * sy,
        bbox[2] * sx,
        bbox[3] * sy,
    ))


def draw_interface(
    frame: np.ndarray,
    frame_id: int,
    first_visible_frame: int,
    threshold: float,
    found: bool,
    detection: dict | None,
    source_frame: np.ndarray,
    pulse: float = 1.0,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    status_color = (70, 210, 80) if found else (0, 190, 255)

    alpha_panel(output, (0, 0), (width, 92), (12, 18, 24), 0.88)
    alpha_panel(
        output, (0, height - 62), (width, height), (12, 18, 24), 0.88)
    text(output, 'RemDet-S  |  VIDEO TARGET SEARCH', (28, 37), 0.82)
    text(output, 'TARGET CLASS: VAN', (29, 73), 0.60, (225, 225, 225), 1)

    status = 'TARGET FOUND' if found else 'SEARCHING...'
    cv2.circle(output, (910, 47), 12, status_color, -1, cv2.LINE_AA)
    text(output, status, (936, 58), 0.86, status_color, 2)

    source_time = max(0, frame_id - 1) / 30.0
    text(
        output,
        f'FRAME {frame_id:03d}   SOURCE TIME {source_time:05.2f}s',
        (28, height - 23),
        0.55,
        (225, 225, 225),
        1,
    )
    text(
        output,
        f'CONF THR {threshold:.2f}   IOU RULE 0.50   FP32',
        (770, height - 23),
        0.50,
        (205, 205, 205),
        1,
    )

    if found and detection is not None:
        x1, y1, x2, y2 = scaled_box(
            detection['bbox'], source_frame.shape, width, height)
        box_color = (50, int(190 + 55 * pulse), 70)
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 5)
        cv2.rectangle(
            output, (max(0, x1), max(94, y1 - 34)),
            (min(width - 1, x1 + 205), max(120, y1)), box_color, -1)
        text(
            output,
            f'VAN  {detection["score"]:.3f}',
            (max(5, x1 + 7), max(116, y1 - 9)),
            0.58,
            (10, 20, 10),
            2,
        )

        # The first visible van is tiny in the 2720x1530 source and enters at
        # the far-left edge.  Add an honest magnified crop without changing
        # the detector box shown on the full frame.
        src_h, src_w = source_frame.shape[:2]
        bx1, by1, bx2, by2 = detection['bbox']
        center_x = (bx1 + bx2) / 2.0
        center_y = (by1 + by2) / 2.0
        crop_w, crop_h = 280, 210
        crop_x1 = int(np.clip(center_x - crop_w / 2, 0, src_w - crop_w))
        crop_y1 = int(np.clip(center_y - crop_h / 2, 0, src_h - crop_h))
        crop = source_frame[
            crop_y1:crop_y1 + crop_h, crop_x1:crop_x1 + crop_w].copy()
        inset_x1, inset_y1 = width - 380, 126
        inset_w, inset_h = 340, 255
        crop = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_CUBIC)
        rel_x1 = int(round((bx1 - crop_x1) / crop_w * inset_w))
        rel_y1 = int(round((by1 - crop_y1) / crop_h * inset_h))
        rel_x2 = int(round((bx2 - crop_x1) / crop_w * inset_w))
        rel_y2 = int(round((by2 - crop_y1) / crop_h * inset_h))
        cv2.rectangle(crop, (rel_x1, rel_y1), (rel_x2, rel_y2), box_color, 3)
        alpha_panel(output, (inset_x1 - 6, inset_y1 - 32),
                    (inset_x1 + inset_w + 6, inset_y1 + inset_h + 6),
                    (8, 12, 16), 0.92)
        text(output, 'DETECTION CLOSE-UP', (inset_x1, inset_y1 - 10),
             0.54, (235, 235, 235), 1)
        output[inset_y1:inset_y1 + inset_h,
               inset_x1:inset_x1 + inset_w] = crop
        cv2.rectangle(
            output, (inset_x1, inset_y1),
            (inset_x1 + inset_w, inset_y1 + inset_h), box_color, 3)

        alpha_panel(output, (width - 380, 405), (width - 40, 570),
                    (8, 12, 16), 0.88)
        text(output, 'DETECTED ON FIRST VISIBLE FRAME',
             (width - 358, 444), 0.54, box_color, 2)
        text(output, f'First visible frame : {first_visible_frame}',
             (width - 358, 483), 0.50, (235, 235, 235), 1)
        text(output, 'Detection delay     : 0 frame',
             (width - 358, 516), 0.50, (235, 235, 235), 1)
        text(output, 'Mission state       : COMPLETE',
             (width - 358, 549), 0.50, (235, 235, 235), 1)

    return output


def title_card(background: np.ndarray, width: int, height: int) -> np.ndarray:
    card = fit_frame(background, width, height)
    alpha_panel(card, (0, 0), (width, height), (5, 10, 16), 0.77)
    text(card, 'RemDet-S', (72, 205), 1.65, (80, 225, 105), 4)
    text(card, 'VIDEO TARGET SEARCH DEMO', (74, 270), 1.10, (255, 255, 255), 2)
    text(card, 'MISSION: FIND A VAN', (76, 350), 0.86, (0, 205, 255), 2)
    text(card, 'VisDrone2019-VID test-dev  |  selected success case',
         (77, 399), 0.59, (220, 220, 220), 1)
    text(card, 'Functional demonstration - not aggregate accuracy',
         (77, 625), 0.52, (175, 185, 195), 1)
    return card


def outro_card(found_frame: np.ndarray) -> np.ndarray:
    card = found_frame.copy()
    height, width = card.shape[:2]
    alpha_panel(card, (0, 0), (width, height), (5, 10, 16), 0.82)
    text(card, 'TARGET ACQUIRED', (70, 280), 1.45, (70, 225, 95), 4)
    text(card, 'Search task completed at the first visible frame.',
         (73, 348), 0.75, (245, 245, 245), 2)
    text(card, 'RemDet-S  |  640x640 model input  |  RTX 5080 FP32',
         (74, 402), 0.60, (205, 210, 215), 1)
    text(card, 'Demo clip selected from the evaluation set.',
         (74, 625), 0.52, (175, 185, 195), 1)
    return card


def load_predictions(path: Path, sequence: str) -> dict[int, list[dict]]:
    output: dict[int, list[dict]] = {}
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if record['sequence'] == sequence:
                output[int(record['frame_id'])] = record['detections']
    return output


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir.resolve()
    predictions_path = args.predictions.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(predictions_path, args.sequence)
    first_detections = [
        item for item in predictions[args.first_visible_frame]
        if item['class_name'] == args.target_class
        and float(item['score']) >= args.threshold
    ]
    earlier = [
        (frame_id, item)
        for frame_id in range(1, args.first_visible_frame)
        for item in predictions.get(frame_id, [])
        if item['class_name'] == args.target_class
        and float(item['score']) >= args.threshold
    ]
    if earlier:
        raise RuntimeError(
            f'Demo is not a clean onset: earlier trigger at {earlier[0][0]}')
    if not first_detections:
        raise RuntimeError('No target detection on the first visible frame')
    detection = max(first_detections, key=lambda item: item['score'])

    frames: dict[int, np.ndarray] = {}
    for frame_id in range(1, args.first_visible_frame + 1):
        path = sequence_dir / f'{frame_id:07d}.jpg'
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f'Cannot read {path}')
        frames[frame_id] = frame

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.output_fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open MP4 writer: {output_path}')

    title = title_card(frames[1], args.width, args.height)
    for _ in range(int(round(args.output_fps * 1.1))):
        writer.write(title)

    # Two output frames per source frame make the pre-onset search leg easy
    # to follow during a presentation while retaining a standard 30 FPS file.
    for frame_id in range(1, args.first_visible_frame):
        fitted = fit_frame(frames[frame_id], args.width, args.height)
        rendered = draw_interface(
            fitted, frame_id, args.first_visible_frame, args.threshold,
            found=False, detection=None, source_frame=frames[frame_id])
        writer.write(rendered)
        writer.write(rendered)

    found_base = fit_frame(
        frames[args.first_visible_frame], args.width, args.height)
    last_found = found_base
    found_duration = int(round(args.output_fps * 2.4))
    for index in range(found_duration):
        pulse = 0.5 + 0.5 * np.sin(index / 5.0)
        last_found = draw_interface(
            found_base,
            args.first_visible_frame,
            args.first_visible_frame,
            args.threshold,
            found=True,
            detection=detection,
            source_frame=frames[args.first_visible_frame],
            pulse=float(pulse),
        )
        writer.write(last_found)

    outro = outro_card(last_found)
    for _ in range(int(round(args.output_fps * 1.2))):
        writer.write(outro)
    writer.release()

    thumbnail_path = output_path.with_suffix('.jpg')
    cv2.imwrite(str(thumbnail_path), last_found)

    capture = cv2.VideoCapture(str(output_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = capture.read()
    capture.release()
    if not ok or frame_count <= 0:
        raise RuntimeError('Generated video could not be decoded')

    metadata = {
        'passed': True,
        'output': str(output_path),
        'thumbnail': str(thumbnail_path),
        'sequence': args.sequence,
        'target_class': args.target_class,
        'confidence_threshold': args.threshold,
        'first_visible_frame': args.first_visible_frame,
        'trigger_frame': args.first_visible_frame,
        'detection_delay_frames': 0,
        'detection': detection,
        'video': {
            'width': video_width,
            'height': video_height,
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': frame_count / fps,
            'size_mb': output_path.stat().st_size / 1024**2,
        },
        'disclosure': (
            'Selected functional success case; not aggregate accuracy.'),
    }
    metadata_path = output_path.with_suffix('.json')
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
