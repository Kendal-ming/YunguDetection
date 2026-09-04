"""Build a reproducible first-appearance manifest from VisDrone2019-VID."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remdet_video.first_appearance.visdrone_vid import (  # noqa: E402
    VISDRONE_CLASSES,
    VidObject,
    find_first_appearance_events,
    index_by_frame,
    read_annotations,
)


DEFAULT_DATASET_ROOT = Path(
    r'C:\Users\xh\Desktop\WindyLab\datasets')
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'work_dirs/first_appearance/manifest'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        '--splits', nargs='+', default=['train', 'val', 'test-dev'])
    parser.add_argument('--target-class', default='van', choices=VISDRONE_CLASSES)
    parser.add_argument('--min-negative-prefix', type=int, default=30)
    parser.add_argument('--min-post-frames', type=int, default=30)
    parser.add_argument('--min-side', type=float, default=8.0)
    parser.add_argument('--max-truncation', type=int, default=1)
    parser.add_argument('--max-occlusion', type=int, default=1)
    parser.add_argument('--persistence-window', type=int, default=5)
    parser.add_argument('--min-persistence', type=int, default=3)
    parser.add_argument('--max-eligibility-delay', type=int, default=30)
    parser.add_argument('--preview-count', type=int, default=12)
    parser.add_argument('--preview-before', type=int, default=2)
    parser.add_argument('--preview-after', type=int, default=2)
    return parser.parse_args()


def frame_paths(sequence_dir: Path) -> list[Path]:
    frames = sorted(sequence_dir.glob('*.jpg'))
    if not frames:
        raise ValueError(f'No JPG frames found in {sequence_dir}')
    expected = [f'{index:07d}' for index in range(1, len(frames) + 1)]
    actual = [frame.stem for frame in frames]
    if actual != expected:
        for index, (actual_name, expected_name) in enumerate(
                zip(actual, expected), start=1):
            if actual_name != expected_name:
                raise ValueError(
                    f'{sequence_dir}: frame {index} is {actual_name}, '
                    f'expected {expected_name}')
        raise ValueError(f'{sequence_dir}: frame numbering is not contiguous')
    zero_size = [frame for frame in frames if frame.stat().st_size == 0]
    if zero_size:
        raise ValueError(f'Zero-size image: {zero_size[0]}')
    return frames


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    fieldnames = list(rows[0])
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def evenly_spaced(items: list[dict], count: int) -> list[dict]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]
    indices = {
        round(index * (len(items) - 1) / (count - 1))
        for index in range(count)
    }
    return [items[index] for index in sorted(indices)]


def choose_previews(events: list[dict], count: int) -> list[dict]:
    by_split: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_split[event['split']].append(event)
    for split_events in by_split.values():
        split_events.sort(
            key=lambda item: (
                item['strict_size_bucket'] or '',
                item['negative_prefix_frames'],
                item['sequence']))

    split_order = [
        split for split in ('val', 'test-dev', 'train') if split in by_split]
    if not split_order:
        return []
    per_split = max(1, math.ceil(count / len(split_order)))
    chosen: list[dict] = []
    for split in split_order:
        chosen.extend(evenly_spaced(by_split[split], per_split))
    return chosen[:count]


def read_frame(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f'Cannot decode image: {path}')
    return frame


def resize_with_padding(
    frame: np.ndarray,
    width: int = 384,
    height: int = 216,
) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized = cv2.resize(
        frame,
        (max(1, round(source_width * scale)),
         max(1, round(source_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def draw_target_boxes(
    frame: np.ndarray,
    frame_objects: list[VidObject],
    category_id: int,
) -> np.ndarray:
    output = frame.copy()
    for item in frame_objects:
        if item.category_id != category_id:
            continue
        x1, y1, x2, y2 = (round(value) for value in item.bbox_xyxy)
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 40), 3)
        cv2.putText(
            output,
            item.class_name,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    return output


def create_preview(
    event: dict,
    output_dir: Path,
    before: int,
    after: int,
) -> Path:
    annotation_path = Path(event['annotation_path'])
    sequence_dir = Path(event['sequence_dir'])
    grouped = index_by_frame(read_annotations(annotation_path))
    onset = int(event['first_visible_frame'])
    first = max(1, onset - before)
    last = min(int(event['frame_count']), onset + after)
    tiles: list[np.ndarray] = []
    for frame_id in range(first, last + 1):
        frame_path = sequence_dir / f'{frame_id:07d}.jpg'
        frame = read_frame(frame_path)
        frame = draw_target_boxes(
            frame, grouped.get(frame_id, []), int(event['category_id']))
        tile = resize_with_padding(frame)
        color = (40, 40, 240) if frame_id == onset else (230, 230, 230)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                      color, 3 if frame_id == onset else 1)
        label = f'frame {frame_id}'
        if frame_id == onset:
            label += '  FIRST VISIBLE'
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(
            tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            color, 1, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.concatenate(tiles, axis=1)
    output_path = output_dir / (
        f"{event['split']}_{event['sequence']}_{event['class_name']}_"
        f"t{onset:07d}.jpg")
    success, encoded = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError(f'Cannot encode preview: {output_path}')
    encoded.tofile(output_path)
    return output_path


def summarize_events(events: list[dict]) -> dict:
    by_split_class: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for split in sorted({event['split'] for event in events}):
        split_events = [event for event in events if event['split'] == split]
        for class_name in VISDRONE_CLASSES:
            rows = [
                event for event in split_events
                if event['class_name'] == class_name]
            by_split_class[split][class_name] = {
                'sequences_containing_class': len(rows),
                'events_with_negative_prefix': sum(
                    event['first_visible_frame'] > 1 for event in rows),
                'strict_candidates': sum(
                    bool(event['qualifies_strict']) for event in rows),
                'practical_candidates': sum(
                    bool(event['qualifies_practical']) for event in rows),
            }
    return dict(by_split_class)


def build_report(summary: dict, target_events: list[dict]) -> str:
    lines = [
        '# VisDrone2019-VID first-appearance manifest',
        '',
        '## Selection contract',
        '',
        f"- Target class: `{summary['settings']['target_class']}`",
        f"- Minimum class-free prefix: "
        f"{summary['settings']['min_negative_prefix']} frames",
        f"- Minimum frames after onset: "
        f"{summary['settings']['min_post_frames']} frames",
        f"- Practical box minimum side: "
        f"{summary['settings']['min_side']} pixels",
        f"- Maximum truncation/occlusion: "
        f"{summary['settings']['max_truncation']} / "
        f"{summary['settings']['max_occlusion']}",
        f"- Persistence: at least "
        f"{summary['settings']['min_persistence']} of "
        f"{summary['settings']['persistence_window']} frames",
        '',
        '## Dataset integrity',
        '',
        '| Split | Sequences | Frames | Annotation rows | Ignored regions | '
        'Others |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for split, item in summary['dataset'].items():
        lines.append(
            f"| {split} | {item['sequences']} | {item['frames']} | "
            f"{item['annotation_rows']} | {item['ignored_rows']} | "
            f"{item['other_rows']} |")

    lines.extend([
        '',
        '## Practical candidates by class',
        '',
        '| Class | Train | Val | Test-dev | Total |',
        '|---|---:|---:|---:|---:|',
    ])
    by_split_class = summary['events_by_split_class']
    for class_name in VISDRONE_CLASSES:
        counts = [
            by_split_class.get(split, {}).get(class_name, {}).get(
                'practical_candidates', 0)
            for split in ('train', 'val', 'test-dev')
        ]
        lines.append(
            f'| {class_name} | {counts[0]} | {counts[1]} | '
            f'{counts[2]} | {sum(counts)} |')

    lines.extend([
        '',
        f"## `{summary['settings']['target_class']}` candidates",
        '',
        f'- Total practical candidates: {len(target_events)}',
        '- Machine-readable rows: `target_candidate_events.csv`',
        '- Visual checks: `previews/`',
        '',
        'The strict onset is the first annotated target box. The practical '
        'onset is the first sufficiently visible box within the configured '
        'delay. Both are retained so later evaluation cannot silently move '
        'the goalpost.',
        '',
    ])
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / 'previews'
    preview_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        'target_class': args.target_class,
        'min_negative_prefix': args.min_negative_prefix,
        'min_post_frames': args.min_post_frames,
        'min_side': args.min_side,
        'max_truncation': args.max_truncation,
        'max_occlusion': args.max_occlusion,
        'persistence_window': args.persistence_window,
        'min_persistence': args.min_persistence,
        'max_eligibility_delay': args.max_eligibility_delay,
    }

    all_events: list[dict] = []
    dataset_summary: dict[str, dict] = {}
    for split in args.splits:
        split_root = dataset_root / f'VisDrone2019-VID-{split}'
        sequences_root = split_root / 'sequences'
        annotations_root = split_root / 'annotations'
        if not sequences_root.is_dir() or not annotations_root.is_dir():
            raise FileNotFoundError(
                f'{split}: expected sequences and annotations under '
                f'{split_root}')

        sequence_dirs = sorted(
            path for path in sequences_root.iterdir() if path.is_dir())
        annotation_paths = {
            path.stem: path for path in annotations_root.glob('*.txt')}
        sequence_names = {path.name for path in sequence_dirs}
        if sequence_names != set(annotation_paths):
            missing = sorted(sequence_names - set(annotation_paths))
            extra = sorted(set(annotation_paths) - sequence_names)
            raise ValueError(
                f'{split}: annotation/sequence mismatch; missing={missing}, '
                f'extra={extra}')

        split_frames = 0
        annotation_rows = 0
        ignored_rows = 0
        other_rows = 0
        for sequence_index, sequence_dir in enumerate(sequence_dirs, start=1):
            frames = frame_paths(sequence_dir)
            objects = read_annotations(annotation_paths[sequence_dir.name])
            split_frames += len(frames)
            annotation_rows += len(objects)
            ignored_rows += sum(item.category_id == 0 for item in objects)
            other_rows += sum(
                item.category_id == len(VISDRONE_CLASSES) + 1
                for item in objects)
            events = find_first_appearance_events(
                objects,
                len(frames),
                min_negative_prefix=args.min_negative_prefix,
                min_post_frames=args.min_post_frames,
                min_side=args.min_side,
                max_truncation=args.max_truncation,
                max_occlusion=args.max_occlusion,
                persistence_window=args.persistence_window,
                min_persistence=args.min_persistence,
                max_eligibility_delay=args.max_eligibility_delay,
            )
            for event in events:
                all_events.append({
                    'event_id': (
                        f"{split}:{sequence_dir.name}:"
                        f"{event['class_name']}"),
                    'split': split,
                    'sequence': sequence_dir.name,
                    'sequence_dir': str(sequence_dir.resolve()),
                    'annotation_path': str(
                        annotation_paths[sequence_dir.name].resolve()),
                    **event,
                })
            if sequence_index % 10 == 0 or sequence_index == len(sequence_dirs):
                print(
                    f'[{split}] {sequence_index}/{len(sequence_dirs)} '
                    f'sequences, {split_frames} frames')

        dataset_summary[split] = {
            'root': str(split_root.resolve()),
            'sequences': len(sequence_dirs),
            'frames': split_frames,
            'annotation_files': len(annotation_paths),
            'annotation_rows': annotation_rows,
            'ignored_rows': ignored_rows,
            'other_rows': other_rows,
        }

    all_events.sort(
        key=lambda item: (
            item['split'], item['sequence'], item['category_id']))
    candidate_events = [
        event for event in all_events if event['qualifies_practical']]
    target_events = [
        event for event in candidate_events
        if event['class_name'] == args.target_class]

    preview_events = choose_previews(target_events, args.preview_count)
    preview_paths = []
    for event in preview_events:
        preview_paths.append(str(create_preview(
            event,
            preview_dir,
            before=args.preview_before,
            after=args.preview_after,
        )))

    target_size_counts = Counter(
        event['strict_size_bucket'] for event in target_events)
    summary = {
        'dataset_root': str(dataset_root),
        'output_dir': str(output_dir),
        'settings': settings,
        'dataset': dataset_summary,
        'all_class_events': len(all_events),
        'practical_candidate_events': len(candidate_events),
        'target_candidate_events': len(target_events),
        'target_strict_size_counts': dict(target_size_counts),
        'events_by_split_class': summarize_events(all_events),
        'preview_paths': preview_paths,
    }

    write_json(output_dir / 'all_events.json', all_events)
    write_csv(output_dir / 'all_events.csv', all_events)
    write_json(output_dir / 'candidate_events.json', candidate_events)
    write_csv(output_dir / 'candidate_events.csv', candidate_events)
    write_json(output_dir / 'target_candidate_events.json', target_events)
    write_csv(output_dir / 'target_candidate_events.csv', target_events)
    write_json(output_dir / 'summary.json', summary)
    (output_dir / 'report.md').write_text(
        build_report(summary, target_events), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
