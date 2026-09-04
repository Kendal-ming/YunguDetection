"""Evaluate Jetson TensorRT predictions with the local COCO evaluator."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / 'work_dirs/deployment/jetson_results'
BASELINE = {
    'bbox_mAP': 0.247,
    'bbox_mAP_50': 0.415,
    'bbox_mAP_75': 0.250,
    'bbox_mAP_s': 0.154,
    'bbox_mAP_m': 0.367,
    'bbox_mAP_l': 0.470,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--annotations', type=Path,
        default=REPO_ROOT.parent /
        'datasets/VisDrone2019-DET-COCO/annotations/'
        'VisDrone2019-DET_val_coco.json')
    parser.add_argument(
        '--predictions', type=Path,
        default=DEFAULT_RESULTS / 'visdrone_val_predictions_trt_fp16.json')
    parser.add_argument(
        '--output', type=Path,
        default=DEFAULT_RESULTS / 'visdrone_val_coco_eval.json')
    parser.add_argument(
        '--tolerance', type=float, default=0.002,
        help='Maximum accepted absolute difference from the PyTorch baseline.')
    return parser.parse_args()


def mean_valid(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else None


def per_category_ap(evaluator: COCOeval, categories: list[dict]) -> list[dict]:
    # precision has shape [IoU, recall, category, area, maxDets].
    precision = evaluator.eval['precision']
    ious = evaluator.params.iouThrs
    index_50 = int(np.argmin(np.abs(ious - 0.50)))
    index_75 = int(np.argmin(np.abs(ious - 0.75)))
    rows = []
    for category_index, category in enumerate(categories):
        all_iou = precision[:, :, category_index, 0, -1]
        at_50 = precision[index_50, :, category_index, 0, -1]
        at_75 = precision[index_75, :, category_index, 0, -1]
        rows.append({
            'category_id': int(category['id']),
            'category_name': category['name'],
            'AP': mean_valid(all_iou),
            'AP50': mean_valid(at_50),
            'AP75': mean_valid(at_75),
        })
    return rows


def main() -> None:
    args = parse_args()
    for path in (args.annotations, args.predictions):
        if not path.is_file():
            raise FileNotFoundError(path)
    predictions = json.loads(args.predictions.read_text(encoding='utf-8'))
    if not isinstance(predictions, list) or not predictions:
        raise RuntimeError('Prediction JSON must be a non-empty COCO result list.')

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        ground_truth = COCO(str(args.annotations))
        detections = ground_truth.loadRes(str(args.predictions))
        evaluator = COCOeval(ground_truth, detections, 'bbox')
        evaluator.params.imgIds = sorted(ground_truth.getImgIds())
        evaluator.params.catIds = sorted(ground_truth.getCatIds())
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    coco_console = captured.getvalue()
    print(coco_console, end='')

    stats = evaluator.stats
    metrics = {
        'bbox_mAP': float(stats[0]),
        'bbox_mAP_50': float(stats[1]),
        'bbox_mAP_75': float(stats[2]),
        'bbox_mAP_s': float(stats[3]),
        'bbox_mAP_m': float(stats[4]),
        'bbox_mAP_l': float(stats[5]),
        'bbox_AR_1': float(stats[6]),
        'bbox_AR_10': float(stats[7]),
        'bbox_AR_100': float(stats[8]),
        'bbox_AR_s': float(stats[9]),
        'bbox_AR_m': float(stats[10]),
        'bbox_AR_l': float(stats[11]),
    }
    differences = {
        name: metrics[name] - value for name, value in BASELINE.items()
    }
    checks = {
        name: abs(differences[name]) <= args.tolerance for name in BASELINE
    }
    categories = sorted(
        ground_truth.dataset['categories'], key=lambda item: item['id'])
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'passed': all(checks.values()),
        'tolerance': args.tolerance,
        'checks': checks,
        'metrics': metrics,
        'windows_pytorch_baseline': BASELINE,
        'difference_jetson_fp16_minus_windows_pytorch': differences,
        'per_category': per_category_ap(evaluator, categories),
        'files': {
            'annotations': str(args.annotations.resolve()),
            'predictions': str(args.predictions.resolve()),
        },
        'prediction_count': len(predictions),
        'coco_console': coco_console,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'passed': report['passed'],
        'metrics': metrics,
        'baseline': BASELINE,
        'difference': differences,
        'per_category': report['per_category'],
        'report': str(args.output.resolve()),
    }, indent=2, ensure_ascii=False))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
