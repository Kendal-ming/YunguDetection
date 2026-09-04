"""Inspect a folder of real videos and generate a contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(r'C:\Users\xh\Desktop\新建文件夹\clip')
DEFAULT_OUTPUT = PROJECT_ROOT / 'work_dirs/real_video_person_demo/inspection'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or '.jpg'
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f'Cannot encode image: {path}')
    encoded.tofile(str(path))


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(input_dir.rglob('*.mp4'))
    if not videos:
        raise FileNotFoundError(f'No MP4 files under {input_dir}')

    cell_width, image_height, header_height = 400, 225, 32
    cell_height = image_height + header_height
    fractions = (0.10, 0.35, 0.60, 0.85)
    sheet = np.zeros(
        (len(videos) * cell_height, len(fractions) * cell_width, 3),
        dtype=np.uint8)
    metadata = []

    for video_index, path in enumerate(videos, start=1):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open video: {path}')
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else None
        samples = []
        row_y = (video_index - 1) * cell_height
        for column, fraction in enumerate(fractions):
            frame_id = min(
                max(0, int(round((frame_count - 1) * fraction))),
                max(0, frame_count - 1),
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot decode {path} frame {frame_id}')
            frame = cv2.resize(
                frame, (cell_width, image_height), interpolation=cv2.INTER_AREA)
            column_x = column * cell_width
            sheet[row_y + header_height:row_y + cell_height,
                  column_x:column_x + cell_width] = frame
            cv2.putText(
                sheet,
                f'VIDEO {video_index}  |  {fraction:.0%}  |  '
                f'{frame_id / fps:.1f}s',
                (column_x + 10, row_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            samples.append({'fraction': fraction, 'frame_id': frame_id + 1})
        capture.release()
        metadata.append({
            'video_id': video_index,
            'path': str(path),
            'relative_path': str(path.relative_to(input_dir)),
            'size_mb': path.stat().st_size / 1024**2,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration_seconds': duration,
            'samples': samples,
        })
        print(
            f'VIDEO {video_index}: {path.name}, {width}x{height}, '
            f'{fps:.3f} FPS, {duration:.1f}s',
            flush=True,
        )

    contact_sheet = output_dir / 'all_videos_contact_sheet.jpg'
    write_image(contact_sheet, sheet)
    report = {
        'input_dir': str(input_dir),
        'video_count': len(metadata),
        'videos': metadata,
        'contact_sheet': str(contact_sheet),
    }
    report_path = output_dir / 'videos.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
