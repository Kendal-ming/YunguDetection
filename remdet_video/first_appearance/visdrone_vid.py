"""Parse VisDrone-VID annotations and identify class-onset events.

The official VID rows contain ten comma-separated fields::

    frame_id, target_id, x, y, width, height,
    score, category_id, truncation, occlusion

Category zero is an ignored region.  Categories one through ten use the
standard VisDrone class order used by RemDet, while category eleven is the
official ``others`` catch-all class and is not evaluated by this model.  The
Task-2 annotations do not provide stable instance identities, so this module
defines an event as the first appearance of a *class* in one video sequence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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
class VidObject:
    """One VisDrone-VID ground-truth row."""

    frame_id: int
    target_id: int
    x: float
    y: float
    width: float
    height: float
    score: float
    category_id: int
    truncation: int
    occlusion: int

    @property
    def class_name(self) -> str:
        if self.category_id == 0:
            return 'ignored-region'
        if self.category_id == len(VISDRONE_CLASSES) + 1:
            return 'others'
        if not 1 <= self.category_id <= len(VISDRONE_CLASSES):
            return f'class-{self.category_id}'
        return VISDRONE_CLASSES[self.category_id - 1]

    @property
    def bbox_xywh(self) -> list[float]:
        return [self.x, self.y, self.width, self.height]

    @property
    def bbox_xyxy(self) -> list[float]:
        return [
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
        ]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_dict(self) -> dict:
        result = asdict(self)
        result['class_name'] = self.class_name
        result['bbox_xywh'] = self.bbox_xywh
        result['bbox_xyxy'] = self.bbox_xyxy
        result['area'] = self.area
        return result


def read_annotations(path: str | Path) -> list[VidObject]:
    """Read one official annotation file and validate its ten fields."""

    annotation_path = Path(path)
    objects: list[VidObject] = []
    with annotation_path.open('r', encoding='utf-8-sig') as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            values = stripped.split(',')
            if len(values) != 10:
                raise ValueError(
                    f'{annotation_path}:{line_number}: expected 10 fields, '
                    f'found {len(values)}')
            try:
                numeric = [float(value) for value in values]
            except ValueError as error:
                raise ValueError(
                    f'{annotation_path}:{line_number}: non-numeric row') \
                    from error

            item = VidObject(
                frame_id=int(numeric[0]),
                target_id=int(numeric[1]),
                x=numeric[2],
                y=numeric[3],
                width=numeric[4],
                height=numeric[5],
                score=numeric[6],
                category_id=int(numeric[7]),
                truncation=int(numeric[8]),
                occlusion=int(numeric[9]),
            )
            if item.frame_id < 1:
                raise ValueError(
                    f'{annotation_path}:{line_number}: frame_id must be >= 1')
            if not 0 <= item.category_id <= len(VISDRONE_CLASSES) + 1:
                raise ValueError(
                    f'{annotation_path}:{line_number}: unexpected category '
                    f'{item.category_id}')
            if item.category_id > 0 and (
                    item.width <= 0 or item.height <= 0):
                raise ValueError(
                    f'{annotation_path}:{line_number}: non-positive box')
            objects.append(item)
    return objects


def index_by_frame(
    objects: Iterable[VidObject],
) -> dict[int, list[VidObject]]:
    """Group annotations by one-based source frame id."""

    grouped: dict[int, list[VidObject]] = defaultdict(list)
    for item in objects:
        grouped[item.frame_id].append(item)
    return dict(grouped)


def _largest(items: Iterable[VidObject]) -> VidObject | None:
    return max(items, key=lambda item: item.area, default=None)


def _size_bucket(area: float | None) -> str | None:
    if area is None:
        return None
    if area < 32.0**2:
        return 'small'
    if area < 96.0**2:
        return 'medium'
    return 'large'


def _is_eligible(
    item: VidObject,
    min_side: float,
    max_truncation: int,
    max_occlusion: int,
) -> bool:
    return (
        min(item.width, item.height) >= min_side
        and item.truncation <= max_truncation
        and item.occlusion <= max_occlusion
    )


def find_first_appearance_events(
    objects: Iterable[VidObject],
    frame_count: int,
    *,
    min_negative_prefix: int = 30,
    min_post_frames: int = 30,
    min_side: float = 8.0,
    max_truncation: int = 1,
    max_occlusion: int = 1,
    persistence_window: int = 5,
    min_persistence: int = 3,
    max_eligibility_delay: int = 30,
) -> list[dict]:
    """Return one class-onset record for every class present in a sequence.

    ``first_visible_frame`` is deliberately strict: the first annotated pixel
    box of that class.  ``first_eligible_frame`` is a practical onset whose
    box is at least ``min_side`` pixels and is not heavily truncated or
    occluded.  A practical candidate must follow the strict onset within
    ``max_eligibility_delay`` frames and persist in at least
    ``min_persistence`` of the next ``persistence_window`` frames.
    """

    if frame_count < 1:
        raise ValueError('frame_count must be positive')
    if persistence_window < 1:
        raise ValueError('persistence_window must be positive')
    if not 1 <= min_persistence <= persistence_window:
        raise ValueError(
            'min_persistence must be between 1 and persistence_window')

    by_class_frame: dict[int, dict[int, list[VidObject]]] = defaultdict(
        lambda: defaultdict(list))
    for item in objects:
        # RemDet is trained/evaluated on categories 1..10.  Category 0
        # (ignored region) and category 11 (others) must not create events.
        if not 1 <= item.category_id <= len(VISDRONE_CLASSES):
            continue
        if item.frame_id > frame_count:
            raise ValueError(
                f'annotation frame {item.frame_id} exceeds {frame_count}')
        by_class_frame[item.category_id][item.frame_id].append(item)

    events: list[dict] = []
    for category_id in sorted(by_class_frame):
        frame_map = by_class_frame[category_id]
        visible_frames = sorted(frame_map)
        strict_frame = visible_frames[0]

        eligible_by_frame = {
            frame_id: [
                item for item in frame_items
                if _is_eligible(
                    item,
                    min_side=min_side,
                    max_truncation=max_truncation,
                    max_occlusion=max_occlusion,
                )
            ]
            for frame_id, frame_items in frame_map.items()
        }
        eligible_frames = sorted(
            frame_id for frame_id, frame_items in eligible_by_frame.items()
            if frame_items)
        eligible_frame = eligible_frames[0] if eligible_frames else None

        strict_end = min(
            frame_count, strict_frame + persistence_window - 1)
        strict_persistence = sum(
            frame_id in frame_map
            for frame_id in range(strict_frame, strict_end + 1))

        eligible_persistence = 0
        if eligible_frame is not None:
            eligible_end = min(
                frame_count, eligible_frame + persistence_window - 1)
            eligible_persistence = sum(
                bool(eligible_by_frame.get(frame_id))
                for frame_id in range(eligible_frame, eligible_end + 1))

        strict_object = _largest(frame_map[strict_frame])
        eligible_object = (
            _largest(eligible_by_frame[eligible_frame])
            if eligible_frame is not None else None)
        eligibility_delay = (
            eligible_frame - strict_frame
            if eligible_frame is not None else None)

        enough_context = (
            strict_frame - 1 >= min_negative_prefix
            and frame_count - strict_frame + 1 >= min_post_frames)
        qualifies_strict = (
            enough_context
            and strict_persistence >= min_persistence
            and strict_object is not None
            and _is_eligible(
                strict_object,
                min_side=min_side,
                max_truncation=max_truncation,
                max_occlusion=max_occlusion,
            )
        )
        qualifies_practical = (
            enough_context
            and eligible_frame is not None
            and eligibility_delay is not None
            and eligibility_delay <= max_eligibility_delay
            and eligible_persistence >= min_persistence)

        event = {
            'category_id': category_id,
            'class_name': VISDRONE_CLASSES[category_id - 1],
            'frame_count': frame_count,
            'first_visible_frame': strict_frame,
            'first_eligible_frame': eligible_frame,
            'eligibility_delay_frames': eligibility_delay,
            'negative_prefix_frames': strict_frame - 1,
            'post_onset_frames': frame_count - strict_frame + 1,
            'strict_persistence_hits': strict_persistence,
            'eligible_persistence_hits': eligible_persistence,
            'strict_box_count': len(frame_map[strict_frame]),
            'strict_largest_bbox_xywh': (
                strict_object.bbox_xywh if strict_object else None),
            'strict_largest_area': (
                strict_object.area if strict_object else None),
            'strict_size_bucket': _size_bucket(
                strict_object.area if strict_object else None),
            'strict_truncation': (
                strict_object.truncation if strict_object else None),
            'strict_occlusion': (
                strict_object.occlusion if strict_object else None),
            'eligible_box_count': (
                len(eligible_by_frame[eligible_frame])
                if eligible_frame is not None else 0),
            'eligible_largest_bbox_xywh': (
                eligible_object.bbox_xywh if eligible_object else None),
            'eligible_largest_area': (
                eligible_object.area if eligible_object else None),
            'eligible_size_bucket': _size_bucket(
                eligible_object.area if eligible_object else None),
            'eligible_truncation': (
                eligible_object.truncation if eligible_object else None),
            'eligible_occlusion': (
                eligible_object.occlusion if eligible_object else None),
            'qualifies_strict': qualifies_strict,
            'qualifies_practical': qualifies_practical,
        }
        events.append(event)
    return events
