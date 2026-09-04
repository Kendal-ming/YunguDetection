"""NumPy post-processing shared by ONNX and TensorRT deployment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from .preprocess import LetterboxMeta
except ImportError:  # Allow the files to run directly on a lean Jetson bundle.
    from preprocess import LetterboxMeta


VISDRONE_CLASSES = (
    'pedestrian',
    'people',
    'bicycle',
    'car',
    'van',
    'truck',
    'tricycle',
    'awning-tricycle',
    'bus',
    'motor',
)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one xyxy box and an array of xyxy boxes."""
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    width_height = np.clip(right_bottom - left_top, 0.0, None)
    intersection = width_height[:, 0] * width_height[:, 1]
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1]))
    boxes_area = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0.0, None)
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-12)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return score-sorted indices kept by per-class greedy NMS."""
    kept: list[int] = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        order = class_indices[np.argsort(-scores[class_indices], kind='stable')]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            remaining = remaining[
                box_iou(boxes[current], boxes[remaining]) <= iou_threshold]
            order = remaining

    kept_array = np.asarray(kept, dtype=np.int64)
    if kept_array.size:
        kept_array = kept_array[
            np.argsort(-scores[kept_array], kind='stable')]
    return kept_array


def restore_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map boxes from letterboxed input coordinates to source coordinates."""
    restored = boxes.astype(np.float32, copy=True)
    top, _, left, _ = meta.pad_param
    scale_x, scale_y = meta.scale_factor
    restored[:, [0, 2]] = (restored[:, [0, 2]] - left) / scale_x
    restored[:, [1, 3]] = (restored[:, [1, 3]] - top) / scale_y
    original_h, original_w = meta.original_shape
    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, original_w)
    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, original_h)
    return restored


def postprocess_detections(
    boxes: np.ndarray,
    class_scores: np.ndarray,
    meta: LetterboxMeta,
    score_threshold: float = 0.001,
    nms_pre: int = 30000,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
    class_names: Sequence[str] = VISDRONE_CLASSES,
) -> list[dict]:
    """Apply multi-label score filtering, class-aware NMS and box restore."""
    boxes = np.asarray(boxes, dtype=np.float32)
    class_scores = np.asarray(class_scores, dtype=np.float32)
    if boxes.ndim == 3:
        boxes = boxes[0]
    if class_scores.ndim == 3:
        class_scores = class_scores[0]
    if boxes.shape != (class_scores.shape[0], 4):
        raise ValueError(
            f'Incompatible boxes {boxes.shape} and scores {class_scores.shape}.')

    box_indices, labels = np.nonzero(class_scores > score_threshold)
    scores = class_scores[box_indices, labels]
    if scores.size == 0:
        return []

    order = np.argsort(-scores, kind='stable')
    if nms_pre > 0:
        order = order[:nms_pre]
    boxes = boxes[box_indices[order]]
    labels = labels[order].astype(np.int64)
    scores = scores[order]

    kept = class_aware_nms(boxes, scores, labels, iou_threshold)
    kept = kept[:max_detections]
    boxes = restore_boxes(boxes[kept], meta)
    labels = labels[kept]
    scores = scores[kept]

    detections = []
    for bbox, score, label in zip(boxes, scores, labels):
        class_id = int(label)
        class_name = (class_names[class_id]
                      if 0 <= class_id < len(class_names)
                      else f'class_{class_id}')
        detections.append({
            'class_id': class_id,
            'class_name': class_name,
            'score': float(score),
            'bbox': [float(value) for value in bbox],
        })
    return detections
