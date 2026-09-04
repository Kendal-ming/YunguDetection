"""A small, measurable inference wrapper around MMDetection's inferencer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import torch

from mmdet.apis import DetInferencer
from mmdet.models.layers import RepDWConv


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


@dataclass(frozen=True)
class Detection:
    """One post-NMS detection in original-image coordinates."""

    class_id: int
    class_name: str
    score: float
    bbox: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


class RemDetDetector:
    """Load RemDet once and expose timed frame-by-frame inference.

    The converted safe checkpoint only contains ``state_dict`` and therefore
    has no dataset metadata. MMDetection otherwise falls back to COCO names,
    which labels VisDrone class id 3 (car) as COCO class id 3 (motorcycle).
    This wrapper always installs the explicit VisDrone class table.
    """

    def __init__(
        self,
        config: str | Path,
        checkpoint: str | Path,
        device: str = 'cuda:0',
        class_names: Sequence[str] = VISDRONE_CLASSES,
        deploy: bool = False,
        precision: str = 'fp32',
    ) -> None:
        self.config = str(Path(config).resolve())
        self.checkpoint = str(Path(checkpoint).resolve())
        self.device = device
        if precision not in {'fp32', 'amp-fp16'}:
            raise ValueError(
                f'Unsupported precision {precision!r}; use fp32 or amp-fp16.')
        if precision == 'amp-fp16' and not device.startswith('cuda'):
            raise ValueError('amp-fp16 requires a CUDA device.')
        self.precision = precision
        self.class_names = tuple(class_names)
        self.inferencer = DetInferencer(
            model=self.config,
            weights=self.checkpoint,
            device=device,
            palette='random',
            show_progress=False,
        )
        self.inferencer.model.dataset_meta['classes'] = self.class_names
        self.model = self.inferencer.model
        self.deploy_modules_converted = 0
        if deploy:
            # RemDet uses RepDWConv, while the repository's stock
            # SwitchToDeployHook only converts RepVGGBlock. Convert the actual
            # RemDet blocks explicitly for the intended single-branch graph.
            for module in list(self.model.modules()):
                if isinstance(module, RepDWConv) and not module.deploy:
                    module.switch_to_deploy()
                    self.deploy_modules_converted += 1
            self.model.eval()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def warmup(self, frame: np.ndarray, iterations: int = 50) -> None:
        """Warm CUDA kernels and memory allocators before measuring latency."""
        for _ in range(max(0, iterations)):
            self.predict(frame, score_threshold=0.001)

    def predict(
        self,
        frame: np.ndarray,
        score_threshold: float = 0.001,
    ) -> tuple[list[Detection], dict[str, float]]:
        """Run one frame and return detections plus separated wall timings.

        ``model_ms`` includes device transfer, neural network forward, box
        decoding and NMS because MMDetection performs these in ``test_step``.
        Splitting those further would require intrusive framework changes and
        is intentionally postponed until deployment experiments.
        """
        preprocess_start = perf_counter()
        preprocessed = self.inferencer.preprocess([frame], batch_size=1)
        _, model_inputs = next(iter(preprocessed))
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0

        use_cuda = torch.cuda.is_available() and self.device.startswith('cuda')
        if use_cuda:
            torch.cuda.synchronize()
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        model_start = perf_counter()
        use_amp = use_cuda and self.precision == 'amp-fp16'
        with torch.autocast(
                device_type='cuda', dtype=torch.float16, enabled=use_amp):
            predictions = self.inferencer.forward(model_inputs)
        if use_cuda:
            gpu_end.record()
            torch.cuda.synchronize()
        model_ms = (perf_counter() - model_start) * 1000.0
        gpu_model_ms = gpu_start.elapsed_time(gpu_end) if use_cuda else model_ms

        filter_start = perf_counter()
        instances = predictions[0].pred_instances.cpu()
        labels = instances.labels.numpy()
        scores = instances.scores.numpy()
        bboxes = instances.bboxes.numpy()

        detections = []
        for label, score, bbox in zip(labels, scores, bboxes):
            if float(score) < score_threshold:
                continue
            class_id = int(label)
            class_name = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else f'class_{class_id}'
            )
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    score=float(score),
                    bbox=[float(value) for value in bbox],
                ))
        filter_ms = (perf_counter() - filter_start) * 1000.0

        timings = {
            'preprocess_ms': preprocess_ms,
            'model_ms': model_ms,
            'gpu_model_ms': gpu_model_ms,
            'filter_ms': filter_ms,
            'pipeline_ms': preprocess_ms + model_ms + filter_ms,
        }
        return detections, timings
