"""Build a 500-frame COCO pre-annotation pack for human-error review."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from remdet_video.tools.build_real_person_demo import alpha_panel, put_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = Path(r'C:\Users\xh\Desktop\新建文件夹\DJI_0042.MP4')
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / 'work_dirs/real_video_person_demo/dji_0042'
    / 'predictions/video_01.jsonl')
DEFAULT_OUTPUT = (
    PROJECT_ROOT / 'work_dirs/real_video_person_demo'
    / 'dji_0042_annotation_pack_500')
QUOTAS = {
    'high_confidence_positive': 100,
    'borderline_positive': 100,
    'transition_unstable': 75,
    'model_negative': 75,
    'uniform_timeline': 150,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=DEFAULT_VIDEO)
    parser.add_argument('--predictions', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--jpeg-quality', type=int, default=90)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            record['max_score'] = max(
                (item['score'] for item in record['human_detections']),
                default=0.0)
            records.append(record)
    return records


def far_enough(index: int, selected: set[int], minimum_gap: int) -> bool:
    if not selected:
        return True
    ordered = sorted(selected)
    position = bisect.bisect_left(ordered, index)
    if position > 0 and index - ordered[position - 1] < minimum_gap:
        return False
    if position < len(ordered) and ordered[position] - index < minimum_gap:
        return False
    return True


def pick_time_diverse(
    candidates: list[int], quota: int, selected: dict[int, str],
    reason: str, total_frames: int,
) -> int:
    """Choose candidates near evenly spaced target times with gap relaxation."""
    candidates = sorted(set(candidates))
    if not candidates or quota <= 0:
        return 0
    added = 0
    targets = np.linspace(0, total_frames - 1, quota, dtype=int).tolist()
    for minimum_gap in (10, 6, 3, 1):
        if added >= quota:
            break
        for target in targets:
            if added >= quota:
                break
            position = bisect.bisect_left(candidates, target)
            left, right = position - 1, position
            chosen = None
            while left >= 0 or right < len(candidates):
                left_distance = (
                    abs(candidates[left] - target) if left >= 0 else float('inf'))
                right_distance = (
                    abs(candidates[right] - target)
                    if right < len(candidates) else float('inf'))
                if left_distance <= right_distance:
                    candidate = candidates[left]
                    left -= 1
                else:
                    candidate = candidates[right]
                    right += 1
                if candidate not in selected and far_enough(
                        candidate, set(selected), minimum_gap):
                    chosen = candidate
                    break
            if chosen is not None:
                selected[chosen] = reason
                added += 1
    return added


def select_frames(records: list[dict], requested_count: int) -> dict[int, str]:
    total = len(records)
    states = [bool(record['human_detections']) for record in records]
    high = [
        index for index, record in enumerate(records)
        if record['max_score'] >= 0.55]
    borderline = [
        index for index, record in enumerate(records)
        if 0.30 <= record['max_score'] < 0.50]
    negative = [
        index for index, record in enumerate(records)
        if not record['human_detections']]
    transitions = []
    for index, state in enumerate(states):
        start = max(0, index - 3)
        end = min(total, index + 4)
        if any(states[nearby] != state for nearby in range(start, end)):
            transitions.append(index)

    selected: dict[int, str] = {}
    candidate_groups = [
        ('high_confidence_positive', high),
        ('borderline_positive', borderline),
        ('transition_unstable', transitions),
        ('model_negative', negative),
        ('uniform_timeline', list(range(total))),
    ]
    scale = requested_count / sum(QUOTAS.values())
    for reason, candidates in candidate_groups:
        quota = int(round(QUOTAS[reason] * scale))
        pick_time_diverse(candidates, quota, selected, reason, total)

    # Guarantee the exact requested size even if one stratum was sparse.
    if len(selected) < requested_count:
        pick_time_diverse(
            list(range(total)), requested_count - len(selected), selected,
            'quota_fill', total)
    if len(selected) > requested_count:
        keep = sorted(selected)[:requested_count]
        selected = {index: selected[index] for index in keep}
    if len(selected) != requested_count:
        raise RuntimeError(
            f'Selected {len(selected)} frames, expected {requested_count}')
    return selected


def encode_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, encoded = cv2.imencode(
        '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    encoded.tofile(str(path))


def make_preview(
    frame: np.ndarray, record: dict, reason: str, display_index: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    preview_width, preview_height = 384, 216
    preview = cv2.resize(
        frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    sx, sy = preview_width / width, preview_height / height
    for detection in record['human_detections']:
        x1, y1, x2, y2 = detection['bbox']
        box = tuple(int(round(value)) for value in (
            x1 * sx, y1 * sy, x2 * sx, y2 * sy))
        cv2.rectangle(
            preview, (box[0], box[1]), (box[2], box[3]), (40, 230, 80), 2)
        put_text(preview, f"{detection['score']:.2f}",
                 (max(1, box[0]), max(15, box[1] - 3)),
                 0.38, (40, 230, 80), 1)
    alpha_panel(preview, (0, 0), (preview_width, 34), (8, 13, 18), 0.86)
    put_text(
        preview,
        f"#{display_index:03d} F{record['source_frame']:04d} "
        f"T{record['source_time_seconds']:06.1f}s "
        f"N{len(record['human_detections'])}",
        (6, 15), 0.35, (245, 245, 245), 1)
    put_text(preview, reason[:28], (6, 29), 0.30, (0, 205, 255), 1)
    return preview


def write_contact_sheet(
    previews: list[np.ndarray], sheet_index: int, path: Path,
) -> None:
    columns, rows = 5, 5
    tile_height, tile_width = previews[0].shape[:2]
    header_height = 52
    sheet = np.full(
        (header_height + rows * tile_height, columns * tile_width, 3),
        (12, 18, 24), dtype=np.uint8)
    start = (sheet_index - 1) * columns * rows + 1
    end = start + len(previews) - 1
    put_text(sheet, f'DJI_0042 PRE-ANNOTATION REVIEW  |  {start:03d}-{end:03d}',
             (18, 34), 0.72, (245, 245, 245), 2)
    for index, preview in enumerate(previews):
        row, column = divmod(index, columns)
        x, y = column * tile_width, header_height + row * tile_height
        sheet[y:y + tile_height, x:x + tile_width] = preview
    encode_jpeg(path, sheet, 90)


def build_readme(stats: dict) -> str:
    return f"""# DJI_0042 误检审核与预标注数据包

这不是已经确认正确的训练标注。所有框都来自当前 RemDet-S 模型，必须人工审核后才能用于评估或训练。

## 数据包内容

- 原视频总帧数：{stats['source_frame_count']}
- 抽取审核帧数：{stats['selected_frame_count']}
- 图像分辨率：{stats['image_width']} x {stats['image_height']}（保持原始4K）
- 预标注类别：person
- images：未画框的原始图像
- annotations/preannotations_coco.json：COCO格式预标注
- review_manifest.csv：帧号、时间、抽样原因和模型结果
- contact_sheets：快速浏览图，绿色框为模型预标注

## 人工审核规则

1. 框中确实是真人：保留，并在需要时调整边界。
2. 框中不是人：删除该框，不要给误检物体增加类别。
3. 画面中有漏掉的人：补画矩形框。
4. 画面中完全没有人：保留该图，但标注应为空。
5. 一张图中有多个人：必须标出所有清楚可见的人，不能只标模型找到的人。
6. 统一使用 person 一个类别，不再区分 pedestrian 和 people。
7. 不要修改图像文件名，否则COCO标注无法匹配。

## 使用CVAT审核

1. 在CVAT中新建一个图像检测任务，标签名称设置为 person。
2. 上传 images 文件夹里的500张JPG图像。
3. 在任务的上传标注功能中选择 COCO 1.0，导入 annotations/preannotations_coco.json。
4. 按上述规则逐张检查；重点检查 review_manifest.csv 中的 borderline_positive、transition_unstable 和 model_negative。
5. 完成后导出 COCO 1.0，文件名建议为 dji_0042_corrected_coco.zip。

也可以尝试直接导入 cvat_dataset_dji0042_500.zip；如果当前CVAT版本不接受整包，就使用上面的“先上传图像、再导入JSON”方式。

## 重要说明

- 绿色预标注框不是标准答案。
- model_negative 表示模型没有检测到人，不代表画面真的没人；这部分专门用于检查漏检。
- 空图片也是重要负样本，不能从数据集中删除。
- 人工修正后的数据应再按不同视频或不同时间段拆分训练集、验证集和测试集，不能把相邻帧随机打散后分组。
"""


def main() -> None:
    args = parse_args()
    video_path = args.video.resolve()
    prediction_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / 'images'
    annotation_dir = output_dir / 'annotations'
    sheet_dir = output_dir / 'contact_sheets'
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(prediction_path)
    selected = select_frames(records, args.count)
    selected_indices = sorted(selected)
    selected_set = set(selected_indices)
    selected_rank = {
        frame_index: rank
        for rank, frame_index in enumerate(selected_indices, start=1)}
    print('Selection reasons:', dict(Counter(selected.values())), flush=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open {video_path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count != len(records):
        raise RuntimeError(
            f'Video has {frame_count} frames but predictions have {len(records)}')

    coco_images = []
    coco_annotations = []
    manifest_rows = []
    annotation_id = 1
    previews: list[np.ndarray] = []
    sheet_index = 1
    extracted = 0
    for frame_index in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {frame_index}')
        if frame_index not in selected_set:
            continue
        record = records[frame_index]
        rank = selected_rank[frame_index]
        reason = selected[frame_index]
        timestamp_ms = int(round(record['source_time_seconds'] * 1000))
        filename = (
            f'DJI_0042_f{record["source_frame"]:06d}_t{timestamp_ms:09d}ms.jpg')
        encode_jpeg(image_dir / filename, frame, args.jpeg_quality)
        image_id = rank
        coco_images.append({
            'id': image_id,
            'file_name': filename,
            'width': width,
            'height': height,
            'frame_index': frame_index,
            'source_frame': record['source_frame'],
            'source_time_seconds': record['source_time_seconds'],
            'selection_reason': reason,
        })
        for detection in record['human_detections']:
            x1, y1, x2, y2 = detection['bbox']
            x1 = float(np.clip(x1, 0, width))
            y1 = float(np.clip(y1, 0, height))
            x2 = float(np.clip(x2, 0, width))
            y2 = float(np.clip(y2, 0, height))
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            if box_width <= 0 or box_height <= 0:
                continue
            coco_annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': [x1, y1, box_width, box_height],
                'area': box_width * box_height,
                'iscrowd': 0,
                'segmentation': [],
                'score': detection['score'],
                'source_class': detection['class_name'],
                'review_status': 'unreviewed_model_preannotation',
            })
            annotation_id += 1
        manifest_rows.append({
            'review_index': rank,
            'filename': filename,
            'source_frame': record['source_frame'],
            'source_time_seconds': f"{record['source_time_seconds']:.3f}",
            'selection_reason': reason,
            'model_box_count': len(record['human_detections']),
            'model_max_score': f"{record['max_score']:.6f}",
            'human_review': '',
            'review_notes': '',
        })
        previews.append(make_preview(frame, record, reason, rank))
        extracted += 1
        if len(previews) == 25:
            write_contact_sheet(
                previews, sheet_index,
                sheet_dir / f'review_sheet_{sheet_index:02d}.jpg')
            previews = []
            sheet_index += 1
        if extracted % 50 == 0:
            print(f'Extracted {extracted}/{args.count} review frames', flush=True)
    capture.release()
    if previews:
        write_contact_sheet(
            previews, sheet_index,
            sheet_dir / f'review_sheet_{sheet_index:02d}.jpg')
    if extracted != args.count:
        raise RuntimeError(f'Extracted {extracted}, expected {args.count}')

    coco = {
        'info': {
            'description': (
                'UNREVIEWED RemDet-S pre-annotations for DJI_0042; '
                'must be corrected by a human before evaluation or training.'),
            'version': '1.0-preannotation',
        },
        'licenses': [],
        'images': coco_images,
        'annotations': coco_annotations,
        'categories': [{'id': 1, 'name': 'person', 'supercategory': 'human'}],
    }
    coco_path = annotation_dir / 'preannotations_coco.json'
    coco_text = json.dumps(coco, ensure_ascii=False, indent=2)
    coco_path.write_text(coco_text, encoding='utf-8')
    (annotation_dir / 'instances_default.json').write_text(
        coco_text, encoding='utf-8')

    manifest_path = output_dir / 'review_manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    stats = {
        'passed': True,
        'source_video': str(video_path),
        'source_predictions': str(prediction_path),
        'source_frame_count': frame_count,
        'source_duration_seconds': frame_count / fps,
        'image_width': width,
        'image_height': height,
        'selected_frame_count': extracted,
        'preannotation_count': len(coco_annotations),
        'selection_reason_counts': dict(Counter(selected.values())),
        'frames_with_preannotations': sum(
            bool(record['human_detections'])
            for index, record in enumerate(records) if index in selected_set),
        'frames_without_preannotations': sum(
            not record['human_detections']
            for index, record in enumerate(records) if index in selected_set),
        'warning': (
            'These are unreviewed model proposals, not ground truth labels.'),
    }
    readme_path = output_dir / 'README_CN.md'
    readme_path.write_text(build_readme(stats), encoding='utf-8')
    stats_path = output_dir / 'dataset_stats.json'
    stats['images_dir'] = str(image_dir)
    stats['coco_preannotations'] = str(coco_path)
    stats['manifest'] = str(manifest_path)
    stats['contact_sheets_dir'] = str(sheet_dir)

    zip_path = output_dir / 'cvat_dataset_dji0042_500.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as archive:
        for image_path in sorted(image_dir.glob('*.jpg')):
            archive.write(image_path, f'images/{image_path.name}')
        archive.write(
            annotation_dir / 'instances_default.json',
            'annotations/instances_default.json')
        archive.write(manifest_path, manifest_path.name)
        archive.write(readme_path, readme_path.name)
    stats['cvat_dataset_zip'] = str(zip_path)
    stats['cvat_dataset_zip_size_mb'] = zip_path.stat().st_size / 1024**2
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
