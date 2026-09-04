"""Evaluate fixed operating thresholds on cached VisDrone predictions.

This is intentionally different from COCO AP: it reports the concrete
precision/recall/F1 obtained by deploying one score threshold. Matching is
class-wise and one-to-one at a chosen IoU threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import mmengine
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        default=[0.10, 0.20, 0.30, 0.40, 0.50],
    )
    parser.add_argument(
        '--iou-thresholds', nargs='+', type=float, default=[0.50, 0.75])
    return parser.parse_args()


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.astype(np.float32, copy=True)
    converted[:, 2] += converted[:, 0]
    converted[:, 3] += converted[:, 1]
    return converted


def pairwise_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection_wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = (
        np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
        * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    )
    union = box_area + boxes_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def match_group(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> tuple[int, int, int]:
    matched = np.zeros(len(gt_boxes), dtype=bool)
    true_positives = 0
    false_positives = 0
    for pred_box in pred_boxes:
        ious = pairwise_iou(pred_box, gt_boxes)
        if len(ious):
            ious[matched] = -1.0
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold:
                matched[best] = True
                true_positives += 1
                continue
        false_positives += 1
    false_negatives = int((~matched).sum())
    return true_positives, false_positives, false_negatives


def metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def as_numpy(value) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_data = json.loads(
        Path(args.annotations).read_text(encoding='utf-8'))
    categories = sorted(annotation_data['categories'], key=lambda item: item['id'])
    category_id_to_index = {
        category['id']: index for index, category in enumerate(categories)
    }
    class_names = [category['name'] for category in categories]

    ground_truth = defaultdict(list)
    for annotation in annotation_data['annotations']:
        if annotation.get('iscrowd', 0) or annotation.get('ignore', 0):
            continue
        class_index = category_id_to_index[annotation['category_id']]
        ground_truth[(annotation['image_id'], class_index)].append(
            annotation['bbox'])
    ground_truth = {
        key: xywh_to_xyxy(np.asarray(boxes, dtype=np.float32))
        for key, boxes in ground_truth.items()
    }

    predictions = mmengine.load(args.predictions)
    prediction_by_image = {item['img_id']: item['pred_instances'] for item in predictions}
    image_ids = [image['id'] for image in annotation_data['images']]

    rows = []
    for iou_threshold in args.iou_thresholds:
        for score_threshold in args.thresholds:
            counts = [dict(tp=0, fp=0, fn=0) for _ in class_names]
            detections_kept = 0
            for image_id in image_ids:
                instances = prediction_by_image[image_id]
                labels = as_numpy(instances['labels']).astype(np.int64)
                scores = as_numpy(instances['scores']).astype(np.float32)
                bboxes = as_numpy(instances['bboxes']).astype(np.float32)
                keep = scores >= score_threshold
                labels = labels[keep]
                scores = scores[keep]
                bboxes = bboxes[keep]
                detections_kept += int(keep.sum())

                for class_index in range(len(class_names)):
                    class_mask = labels == class_index
                    order = np.argsort(-scores[class_mask])
                    pred_boxes = bboxes[class_mask][order]
                    gt_boxes = ground_truth.get(
                        (image_id, class_index),
                        np.empty((0, 4), dtype=np.float32),
                    )
                    tp, fp, fn = match_group(
                        pred_boxes, gt_boxes, iou_threshold)
                    counts[class_index]['tp'] += tp
                    counts[class_index]['fp'] += fp
                    counts[class_index]['fn'] += fn

            for class_index, class_name in enumerate(class_names):
                count = counts[class_index]
                row_metrics = metrics(count['tp'], count['fp'], count['fn'])
                rows.append({
                    'scope': 'class',
                    'class_name': class_name,
                    'score_threshold': score_threshold,
                    'iou_threshold': iou_threshold,
                    **count,
                    **row_metrics,
                    'detections_per_image': detections_kept / len(image_ids),
                })

            total = {
                key: sum(count[key] for count in counts)
                for key in ('tp', 'fp', 'fn')
            }
            micro = metrics(total['tp'], total['fp'], total['fn'])
            class_metrics = [
                metrics(count['tp'], count['fp'], count['fn'])
                for count in counts
            ]
            rows.append({
                'scope': 'micro',
                'class_name': 'all',
                'score_threshold': score_threshold,
                'iou_threshold': iou_threshold,
                **total,
                **micro,
                'detections_per_image': detections_kept / len(image_ids),
            })
            rows.append({
                'scope': 'macro',
                'class_name': 'all',
                'score_threshold': score_threshold,
                'iou_threshold': iou_threshold,
                **total,
                'precision': float(np.mean([
                    item['precision'] for item in class_metrics
                ])),
                'recall': float(np.mean([
                    item['recall'] for item in class_metrics
                ])),
                'f1': float(np.mean([item['f1'] for item in class_metrics])),
                'detections_per_image': detections_kept / len(image_ids),
            })

    with (output_dir / 'threshold_metrics.csv').open(
            'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'threshold_metrics.json').open(
            'w', encoding='utf-8') as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    print(f'wrote {len(rows)} rows to {output_dir}')


if __name__ == '__main__':
    main()
