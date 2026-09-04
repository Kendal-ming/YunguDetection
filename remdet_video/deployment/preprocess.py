"""Framework-independent preprocessing matching the RemDet test pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry needed to map 640x640 predictions to the source image."""

    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    input_shape: tuple[int, int]
    scale_factor: tuple[float, float]
    pad_param: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        return asdict(self)


def preprocess_bgr(
    image: np.ndarray,
    input_size: int | tuple[int, int] = 640,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize, letterbox, BGR->RGB and normalize an OpenCV image.

    The implementation mirrors the configured ``YOLOv5KeepRatioResize`` +
    ``LetterResize`` + ``YOLOv5DetDataPreprocessor`` path. The returned array
    is contiguous FP32 NCHW with a batch dimension.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a non-empty BGR image with shape HxWx3.')

    if isinstance(input_size, int):
        target_h = target_w = input_size
    else:
        target_h, target_w = input_size
    if target_h <= 0 or target_w <= 0:
        raise ValueError('Input dimensions must be positive.')

    original_h, original_w = image.shape[:2]
    ratio = min(target_h / original_h, target_w / original_w)

    # YOLOv5KeepRatioResize uses int() rather than round() for this stage.
    resized_w = int(original_w * ratio)
    resized_h = int(original_h * ratio)
    if (resized_w, resized_h) != (original_w, original_h):
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(
            image, (resized_w, resized_h), interpolation=interpolation)
    else:
        resized = image

    padding_h = target_h - resized_h
    padding_w = target_w - resized_w
    top = int(round(padding_h // 2 - 0.1))
    left = int(round(padding_w // 2 - 0.1))
    bottom = padding_h - top
    right = padding_w - left
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    rgb_chw = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(rgb_chw, dtype=np.float32)[None] / 255.0
    meta = LetterboxMeta(
        original_shape=(original_h, original_w),
        resized_shape=(resized_h, resized_w),
        input_shape=(target_h, target_w),
        scale_factor=(resized_w / original_w, resized_h / original_h),
        pad_param=(top, bottom, left, right),
    )
    return tensor, meta
