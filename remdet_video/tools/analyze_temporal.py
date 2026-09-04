"""Apply threshold and temporal-rule sweeps to cached video detections."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remdet_video.core.temporal_filter import (  # noqa: E402
    DEFAULT_TEMPORAL_RULES,
    apply_temporal_rule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-jsonl', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--target-classes',
        nargs='+',
        default=['car', 'van', 'truck', 'bus', 'motor'],
    )
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    return parser.parse_args()


def count_transitions(sequence: list[bool]) -> int:
    return sum(left != right for left, right in zip(sequence, sequence[1:]))


def count_events(sequence: list[bool]) -> int:
    previous = False
    events = 0
    for value in sequence:
        if value and not previous:
            events += 1
        previous = value
    return events


def positive_runs(sequence: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for index, value in enumerate(sequence + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [
        json.loads(line)
        for line in Path(args.frames_jsonl).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]

    rows = []
    timelines = {}
    target_sets = [('all_targets', set(args.target_classes))]
    target_sets.extend((name, {name}) for name in args.target_classes)

    for target_name, target_classes in target_sets:
        for threshold in args.thresholds:
            raw_presence = [
                any(
                    detection['class_name'] in target_classes
                    and detection['score'] >= threshold
                    for detection in frame['detections'])
                for frame in frames
            ]
            raw_runs = positive_runs(raw_presence)
            for rule in DEFAULT_TEMPORAL_RULES:
                benchmark_repeats = 1000
                benchmark_start = perf_counter()
                for _ in range(benchmark_repeats):
                    filtered = apply_temporal_rule(raw_presence, rule)
                temporal_us_per_frame = (
                    (perf_counter() - benchmark_start) * 1_000_000.0
                    / (benchmark_repeats * len(raw_presence))
                )
                filtered_runs = positive_runs(filtered)
                key = f'{target_name}|{threshold:.2f}|{rule.name}'
                timelines[key] = {
                    'raw': raw_presence,
                    'filtered': filtered,
                }
                entry_delays = []
                for raw_start, raw_end in raw_runs:
                    detections = [
                        index for index in range(raw_start, raw_end + 1)
                        if filtered[index]
                    ]
                    if detections:
                        entry_delays.append(detections[0] - raw_start)
                rows.append({
                    'target': target_name,
                    'threshold': threshold,
                    'rule': rule.name,
                    'frames': len(frames),
                    'raw_positive_frames': sum(raw_presence),
                    'filtered_positive_frames': sum(filtered),
                    'raw_events': count_events(raw_presence),
                    'filtered_events': count_events(filtered),
                    'raw_transitions': count_transitions(raw_presence),
                    'filtered_transitions': count_transitions(filtered),
                    'raw_isolated_events': sum(
                        end - start + 1 <= 1 for start, end in raw_runs),
                    'filtered_isolated_events': sum(
                        end - start + 1 <= 1 for start, end in filtered_runs),
                    'mean_proxy_entry_delay_frames': (
                        sum(entry_delays) / len(entry_delays)
                        if entry_delays else None),
                    'temporal_us_per_frame': temporal_us_per_frame,
                })

    fieldnames = list(rows[0]) if rows else []
    with (output_dir / 'temporal_sweep.csv').open(
            'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'temporal_timelines.json').open(
            'w', encoding='utf-8') as file:
        json.dump(timelines, file, ensure_ascii=False)

    print(f'wrote {len(rows)} temporal sweep rows to {output_dir}')


if __name__ == '__main__':
    main()
