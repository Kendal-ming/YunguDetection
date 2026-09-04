"""A small ONNX-friendly RemDet inference graph."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class RemDetONNXWrapper(nn.Module):
    """Expose decoded boxes and class probabilities without framework NMS.

    Keeping thresholding and NMS outside the graph avoids custom TensorRT
    plugins and lets deployment experiments vary those settings freely.
    """

    def __init__(
        self,
        detector: nn.Module,
        input_size: int = 640,
        strides: Sequence[int] = (8, 16, 32),
    ) -> None:
        super().__init__()
        self.detector = detector
        self.input_size = int(input_size)
        self.strides = tuple(int(value) for value in strides)

        points = []
        stride_values = []
        for stride in self.strides:
            if self.input_size % stride != 0:
                raise ValueError(
                    f'Input size {self.input_size} is not divisible by {stride}.')
            size = self.input_size // stride
            y, x = torch.meshgrid(
                torch.arange(size, dtype=torch.float32),
                torch.arange(size, dtype=torch.float32),
                indexing='ij',
            )
            points.append(torch.stack((x + 0.5, y + 0.5), dim=-1).reshape(-1, 2)
                          * stride)
            stride_values.append(torch.full((size * size, 1), float(stride)))

        self.register_buffer('points', torch.cat(points, dim=0), persistent=True)
        self.register_buffer(
            'stride_values', torch.cat(stride_values, dim=0), persistent=True)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.detector.extract_feat(images)
        cls_scores, bbox_distances = self.detector.bbox_head(features)

        scores = torch.cat([
            score.permute(0, 2, 3, 1).reshape(images.shape[0], -1,
                                               score.shape[1])
            for score in cls_scores
        ], dim=1).sigmoid()
        distances = torch.cat([
            distance.permute(0, 2, 3, 1).reshape(images.shape[0], -1, 4)
            for distance in bbox_distances
        ], dim=1) * self.stride_values.unsqueeze(0)

        point_x = self.points[:, 0].unsqueeze(0)
        point_y = self.points[:, 1].unsqueeze(0)
        boxes = torch.stack((
            point_x - distances[..., 0],
            point_y - distances[..., 1],
            point_x + distances[..., 2],
            point_y + distances[..., 3],
        ), dim=-1)
        return boxes, scores
