"""Evaluate false triggers and first-detection delay from cached predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from remdet_video.first_appearance.visdrone_vid import (  # noqa: E402
    index_by_frame,
    read_annotations,
)


DEFAULT_EVENTS = (
    PROJECT_ROOT
    / 'work_dirs/first_appearance/manifest/target_candidate_events.json')
DEFAULT_INFERENCE = (
    PROJECT_ROOT / 'work_dirs/first_appearance/inference/remdet_s_fp32')
DEFAULT_OUTPUT = (
    PROJECT_ROOT / 'work_dirs/first_appearance/evaluation/remdet_s_fp32')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, default=DEFAULT_EVENTS)
    parser.add_argument('--inference-dir', type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--thresholds', type=float, nargs='+',
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80])
    parser.add_argument('--iou-threshold', type=float, default=0.50)
    parser.add_argument('--max-delay-frames', type=int, default=30)
    parser.add_argument('--source-fps', type=float, default=30.0)
    parser.add_argument('--development-split', default='train')
    parser.add_argument('--test-split', default='test-dev')
    return parser.parse_args()


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = (
        max(0.0, right[2] - right[0])
        * max(0.0, right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def wilson(successes: int, total: int, z: float = 1.96) -> list[float | None]:
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def load_predictions(path: Path) -> dict[tuple[str, str], dict[int, list[dict]]]:
    output: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(dict)
    with path.open('r', encoding='utf-8') as stream:
        for raw_line in stream:
            record = json.loads(raw_line)
            key = (record['split'], record['sequence'])
            output[key][int(record['frame_id'])] = record['detections']
    return dict(output)


def is_correct(
    detections: list[dict],
    gt_boxes: list[list[float]],
    threshold: float,
    iou_threshold: float,
) -> bool:
    return any(
        detection['score'] >= threshold
        and any(
            bbox_iou(detection['bbox'], gt_box) >= iou_threshold
            for gt_box in gt_boxes)
        for detection in detections
    )


def maximum_correct_score(
    detections: list[dict],
    gt_boxes: list[list[float]],
    iou_threshold: float,
) -> float:
    matching_scores = [
        float(detection['score'])
        for detection in detections
        if any(
            bbox_iou(detection['bbox'], gt_box) >= iou_threshold
            for gt_box in gt_boxes)
    ]
    return max(matching_scores, default=0.0)


def evaluate_event(
    event: dict,
    frame_predictions: dict[int, list[dict]],
    gt_by_frame: dict[int, list],
    threshold: float,
    iou_threshold: float,
    max_delay_frames: int,
    source_fps: float,
) -> dict:
    category_id = int(event['category_id'])
    class_name = event['class_name']
    onset = int(event['first_visible_frame'])
    eligible = int(event['first_eligible_frame'])
    final_frame = min(int(event['frame_count']), onset + max_delay_frames)

    def predictions(frame_id: int) -> list[dict]:
        return [
            item for item in frame_predictions.get(frame_id, [])
            if item['class_name'] == class_name and item['score'] >= threshold
        ]

    def gt_boxes(frame_id: int) -> list[list[float]]:
        return [
            item.bbox_xyxy for item in gt_by_frame.get(frame_id, [])
            if item.category_id == category_id
        ]

    raw_pre_scores = [
        float(item['score'])
        for frame_id in range(1, onset)
        for item in frame_predictions.get(frame_id, [])
        if item['class_name'] == class_name
    ]
    maximum_pre_target_score = max(raw_pre_scores, default=0.0)
    raw_onset_detections = [
        item for item in frame_predictions.get(onset, [])
        if item['class_name'] == class_name
    ]
    maximum_correct_onset_score = maximum_correct_score(
        raw_onset_detections, gt_boxes(onset), iou_threshold)
    score_separation_margin = (
        maximum_correct_onset_score - maximum_pre_target_score)

    pre_trigger_frames = [
        frame_id for frame_id in range(1, onset)
        if predictions(frame_id)
    ]
    pre_trigger_detection_count = sum(
        len(predictions(frame_id)) for frame_id in range(1, onset))
    strict_hit = is_correct(
        predictions(onset), gt_boxes(onset), threshold,
        iou_threshold)
    practical_hit = is_correct(
        predictions(eligible), gt_boxes(eligible), threshold,
        iou_threshold)

    first_trigger_frame = next((
        frame_id for frame_id in range(1, final_frame + 1)
        if predictions(frame_id)
    ), None)
    first_correct_frame = next((
        frame_id for frame_id in range(onset, final_frame + 1)
        if is_correct(
            predictions(frame_id), gt_boxes(frame_id), threshold,
            iou_threshold)
    ), None)
    first_trigger_correct = (
        first_trigger_frame is not None
        and first_trigger_frame >= onset
        and is_correct(
            predictions(first_trigger_frame), gt_boxes(first_trigger_frame),
            threshold, iou_threshold)
    )
    delay = (
        first_correct_frame - onset if first_correct_frame is not None else None)

    return {
        'event_id': event['event_id'],
        'split': event['split'],
        'sequence': event['sequence'],
        'class_name': class_name,
        'threshold': threshold,
        'first_visible_frame': onset,
        'first_eligible_frame': eligible,
        'negative_prefix_frames': onset - 1,
        'strict_size_bucket': event['strict_size_bucket'],
        'maximum_pre_target_score': maximum_pre_target_score,
        'maximum_correct_onset_score': maximum_correct_onset_score,
        'score_separation_margin': score_separation_margin,
        'threshold_separable': score_separation_margin > 0.0,
        'pre_target_false_trigger': bool(pre_trigger_frames),
        'pre_target_trigger_frame_count': len(pre_trigger_frames),
        'pre_target_detection_count': pre_trigger_detection_count,
        'first_pre_target_trigger_frame': (
            pre_trigger_frames[0] if pre_trigger_frames else None),
        'strict_first_frame_hit': strict_hit,
        'practical_first_frame_hit': practical_hit,
        'first_trigger_frame': first_trigger_frame,
        'first_trigger_correct': first_trigger_correct,
        'first_correct_frame': first_correct_frame,
        'detection_delay_frames': delay,
        'detection_delay_ms_at_source_fps': (
            delay / source_fps * 1000.0 if delay is not None else None),
        'strict_task_success': not pre_trigger_frames and strict_hit,
        'practical_task_success': not pre_trigger_frames and practical_hit,
        'correct_first_trigger_within_1_frame': (
            first_trigger_correct and first_trigger_frame <= onset + 1),
        'correct_first_trigger_within_3_frames': (
            first_trigger_correct and first_trigger_frame <= onset + 3),
        'correct_first_trigger_within_5_frames': (
            first_trigger_correct and first_trigger_frame <= onset + 5),
        'correct_first_trigger_within_30_frames': (
            first_trigger_correct
            and first_trigger_frame <= onset + max_delay_frames),
    }


def aggregate(rows: list[dict], split: str, threshold: float) -> dict:
    selected = [
        row for row in rows
        if row['split'] == split and row['threshold'] == threshold
    ]
    total = len(selected)

    def count(name: str) -> int:
        return sum(bool(row[name]) for row in selected)

    def rate(name: str) -> float | None:
        return count(name) / total if total else None

    delay_values = [
        row['detection_delay_frames'] for row in selected
        if not row['pre_target_false_trigger']
        and row['detection_delay_frames'] is not None
    ]
    pre_frames = sum(row['negative_prefix_frames'] for row in selected)
    pre_trigger_frames = sum(
        row['pre_target_trigger_frame_count'] for row in selected)
    strict_successes = count('strict_task_success')
    separable_events = count('threshold_separable')
    return {
        'split': split,
        'threshold': threshold,
        'event_count': total,
        'negative_prefix_frames': pre_frames,
        'false_trigger_events': count('pre_target_false_trigger'),
        'false_trigger_event_rate': rate('pre_target_false_trigger'),
        'false_trigger_frame_rate': (
            pre_trigger_frames / pre_frames if pre_frames else None),
        'strict_first_frame_recall': rate('strict_first_frame_hit'),
        'practical_first_frame_recall': rate('practical_first_frame_hit'),
        'strict_task_successes': strict_successes,
        'strict_task_success_rate': rate('strict_task_success'),
        'strict_task_success_wilson95': wilson(strict_successes, total),
        'threshold_separable_events': separable_events,
        'threshold_separable_event_rate': (
            separable_events / total if total else None),
        'practical_task_success_rate': rate('practical_task_success'),
        'correct_first_trigger_within_1_frame_rate': rate(
            'correct_first_trigger_within_1_frame'),
        'correct_first_trigger_within_3_frames_rate': rate(
            'correct_first_trigger_within_3_frames'),
        'correct_first_trigger_within_5_frames_rate': rate(
            'correct_first_trigger_within_5_frames'),
        'correct_first_trigger_within_30_frames_rate': rate(
            'correct_first_trigger_within_30_frames'),
        'safe_first_correct_delay_mean_frames': (
            mean(delay_values) if delay_values else None),
        'safe_first_correct_delay_median_frames': (
            median(delay_values) if delay_values else None),
    }


def choose_threshold(aggregates: list[dict], development_split: str) -> float:
    candidates = [
        row for row in aggregates
        if row['split'] == development_split and row['event_count'] > 0
    ]
    zero_false = [row for row in candidates if row['false_trigger_events'] == 0]
    pool = zero_false or candidates
    if not pool:
        raise ValueError(f'No development rows for split {development_split}')
    # Maximize complete task success.  Ties prefer first-frame recall, then a
    # lower threshold (more margin against a missed first appearance).
    selected = max(
        pool,
        key=lambda row: (
            row['strict_task_success_rate'],
            row['strict_first_frame_recall'],
            -row['threshold'],
        ),
    )
    return float(selected['threshold'])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_tradeoff(path: Path, aggregates: list[dict]) -> None:
    splits = sorted({row['split'] for row in aggregates})
    figure, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.5))
    if len(splits) == 1:
        axes = [axes]
    for axis, split in zip(axes, splits):
        rows = sorted(
            (row for row in aggregates if row['split'] == split),
            key=lambda row: row['threshold'])
        x = [row['threshold'] for row in rows]
        axis.plot(
            x, [row['strict_first_frame_recall'] for row in rows],
            marker='o', label='first-frame recall')
        axis.plot(
            x, [row['false_trigger_event_rate'] for row in rows],
            marker='s', label='pre-onset false-trigger rate')
        axis.plot(
            x, [row['strict_task_success_rate'] for row in rows],
            marker='^', linewidth=2.5, label='strict task success')
        axis.set_title(f'{split} (n={rows[0]["event_count"]})')
        axis.set_xlabel('confidence threshold')
        axis.set_ylabel('event rate')
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle('First-appearance trade-off (IoU >= 0.50)')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_score_separation(path: Path, rows: list[dict], test_split: str) -> None:
    if not rows:
        return
    first_threshold = min(row['threshold'] for row in rows)
    selected = [
        row for row in rows
        if row['split'] == test_split and row['threshold'] == first_threshold
    ]
    selected.sort(key=lambda row: row['score_separation_margin'])
    if not selected:
        return
    figure, axis = plt.subplots(
        figsize=(10, max(4.5, len(selected) * 0.32)))
    for index, row in enumerate(selected):
        pre_score = row['maximum_pre_target_score']
        onset_score = row['maximum_correct_onset_score']
        color = '#2ca02c' if row['threshold_separable'] else '#d62728'
        axis.plot(
            [pre_score, onset_score], [index, index], color=color,
            linewidth=2, alpha=0.75)
    axis.scatter(
        [row['maximum_pre_target_score'] for row in selected],
        range(len(selected)), marker='x', s=45, color='#d62728',
        label='highest pre-onset target-class score')
    axis.scatter(
        [row['maximum_correct_onset_score'] for row in selected],
        range(len(selected)), marker='o', s=35, color='#1f77b4',
        label='best correct score on first frame')
    axis.set_yticks(range(len(selected)))
    axis.set_yticklabels([
        f'{row["class_name"]} | {row["sequence"]}' for row in selected
    ], fontsize=8)
    axis.set_xlim(-0.02, 1.02)
    axis.set_xlabel('confidence score')
    axis.set_title(
        f'Can one threshold separate false alarms from onset? ({test_split})')
    axis.grid(axis='x', alpha=0.25)
    axis.legend(loc='lower right', fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(
    path: Path,
    aggregates: list[dict],
    chosen_threshold: float,
    development_split: str,
    test_split: str,
    class_counts: dict,
    operating_point_feasible: bool,
) -> None:
    lines = [
        '# RemDet-S first-appearance evaluation',
        '',
        '## Contract',
        '',
        '- A false trigger means the target class was predicted before its '
        'first ground-truth appearance.',
        '- Strict success means no earlier false trigger and an IoU >= 0.50 '
        'match on the exact first visible frame.',
        '- The confidence threshold is selected on the development split and '
        'then frozen for the test split.',
        '',
        f'- Safety-reference threshold: **{chosen_threshold:.2f}** '
        f'(development split: `{development_split}`)',
        f'- Viable strict operating point found: '
        f'**{"yes" if operating_point_feasible else "no"}**',
        f'- Event classes: `{json.dumps(class_counts, ensure_ascii=False)}`',
        '',
        '## Threshold sweep',
        '',
        '| Split | Threshold | N | False-trigger events | First-frame recall | '
        'Strict success | <=3-frame correct trigger | Separable |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in aggregates:
        lines.append(
            f'| {row["split"]} | {row["threshold"]:.2f} | '
            f'{row["event_count"]} | '
            f'{row["false_trigger_events"]} '
            f'({row["false_trigger_event_rate"]:.1%}) | '
            f'{row["strict_first_frame_recall"]:.1%} | '
            f'{row["strict_task_success_rate"]:.1%} | '
            f'{row["correct_first_trigger_within_3_frames_rate"]:.1%} | '
            f'{row["threshold_separable_event_rate"]:.1%} |')

    lines.extend(['', '## Frozen-threshold result', ''])
    if not operating_point_feasible:
        lines.extend([
            '> No tested threshold achieved even one strict development-set '
            'success. The safety-reference threshold below removes observed '
            'pre-onset triggers but is **not** a usable operating point because '
            'it also misses the first visible frame.',
            '',
        ])
    for split in (development_split, test_split):
        row = next((
            item for item in aggregates
            if item['split'] == split
            and item['threshold'] == chosen_threshold), None)
        if row is None:
            continue
        low, high = row['strict_task_success_wilson95']
        lines.extend([
            f'### {split}',
            '',
            f'- Strict task success: {row["strict_task_successes"]}/'
            f'{row["event_count"]} ({row["strict_task_success_rate"]:.1%})',
            f'- 95% Wilson interval: {low:.1%} to {high:.1%}',
            f'- Events with a pre-onset false trigger: '
            f'{row["false_trigger_events"]}/{row["event_count"]}',
            f'- First-visible-frame recall alone: '
            f'{row["strict_first_frame_recall"]:.1%}',
            '',
        ])
    lines.extend([
        '## Important limitation',
        '',
        'This experiment evaluates VisDrone object classes, not color.  A '
        '`van` prediction means any annotated van; it does not mean a yellow '
        'van.  The test sample is also small, so the confidence interval must '
        'be reported with the point estimate.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    events = json.loads(args.events.resolve().read_text(encoding='utf-8'))
    inference_dir = args.inference_dir.resolve()
    metadata = json.loads(
        (inference_dir / 'metadata.json').read_text(encoding='utf-8'))
    predictions = load_predictions(inference_dir / 'predictions.jsonl')
    available_splits = set(metadata['splits'])
    events = [event for event in events if event['split'] in available_splits]
    if not events:
        raise ValueError('No events match the cached inference splits')

    gt_cache: dict[str, dict[int, list]] = {}
    all_rows: list[dict] = []
    for event in events:
        annotation_path = event['annotation_path']
        if annotation_path not in gt_cache:
            gt_cache[annotation_path] = index_by_frame(
                read_annotations(annotation_path))
        key = (event['split'], event['sequence'])
        if key not in predictions:
            raise KeyError(f'Missing cached predictions for {key}')
        for threshold in sorted(set(args.thresholds)):
            all_rows.append(evaluate_event(
                event=event,
                frame_predictions=predictions[key],
                gt_by_frame=gt_cache[annotation_path],
                threshold=threshold,
                iou_threshold=args.iou_threshold,
                max_delay_frames=args.max_delay_frames,
                source_fps=args.source_fps,
            ))

    splits = sorted({event['split'] for event in events})
    aggregates = [
        aggregate(all_rows, split, threshold)
        for split in splits
        for threshold in sorted(set(args.thresholds))
    ]
    chosen_threshold = choose_threshold(aggregates, args.development_split)
    operating_point_feasible = any(
        row['split'] == args.development_split
        and row['strict_task_successes'] > 0
        for row in aggregates)
    class_counts = dict(Counter(event['class_name'] for event in events))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'event_results.csv', all_rows)
    write_csv(output_dir / 'threshold_sweep.csv', aggregates)
    plot_tradeoff(output_dir / 'threshold_tradeoff.png', aggregates)
    plot_score_separation(
        output_dir / 'score_separation.png', all_rows, args.test_split)
    render_report(
        output_dir / 'report.md', aggregates, chosen_threshold,
        args.development_split, args.test_split, class_counts,
        operating_point_feasible)

    chosen_rows = [
        row for row in all_rows if row['threshold'] == chosen_threshold]
    result = {
        'passed': True,
        'events': str(args.events.resolve()),
        'inference_metadata': str(inference_dir / 'metadata.json'),
        'iou_threshold': args.iou_threshold,
        'thresholds': sorted(set(args.thresholds)),
        'development_split': args.development_split,
        'test_split': args.test_split,
        'selected_threshold': chosen_threshold,
        'operating_point_feasible': operating_point_feasible,
        'event_count': len(events),
        'class_counts': class_counts,
        'selected_threshold_by_split': {
            split: next(
                row for row in aggregates
                if row['split'] == split
                and row['threshold'] == chosen_threshold)
            for split in splits
        },
        'selected_threshold_events': chosen_rows,
        'artifacts': {
            'event_results_csv': str(output_dir / 'event_results.csv'),
            'threshold_sweep_csv': str(output_dir / 'threshold_sweep.csv'),
            'tradeoff_plot': str(output_dir / 'threshold_tradeoff.png'),
            'score_separation_plot': str(
                output_dir / 'score_separation.png'),
            'report': str(output_dir / 'report.md'),
        },
    }
    (output_dir / 'result.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
