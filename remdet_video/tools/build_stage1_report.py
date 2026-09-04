"""Aggregate E5/E6 measurements and generate tables, charts and a report."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = PROJECT_ROOT / 'work_dirs' / 'video_experiments' / 'stage1'
OUTPUT_DIR = STAGE1_DIR / 'report'

METRIC_KEYS = {
    'map': 'coco/bbox_mAP',
    'ap50': 'coco/bbox_mAP_50',
    'ap75': 'coco/bbox_mAP_75',
    'aps': 'coco/bbox_mAP_s',
    'apm': 'coco/bbox_mAP_m',
    'apl': 'coco/bbox_mAP_l',
}


def load_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def latest_metric_record(directory: Path) -> dict:
    candidates = sorted(
        path for path in directory.rglob('*.json')
        if path.parent.name.startswith('20'))
    if not candidates:
        raise FileNotFoundError(f'No metric JSON found under {directory}')
    records = []
    with candidates[-1].open(encoding='utf-8') as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    record = next(
        item for item in reversed(records)
        if 'coco/bbox_mAP' in item)
    return {name: float(record[key]) for name, key in METRIC_KEYS.items()}


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trial_row(summary: dict, trial: int, model: str) -> dict:
    latency = summary['latency']
    return {
        'model': model,
        'input_size': int(summary['input_size']),
        'precision': summary.get('precision', 'fp32'),
        'trial': trial,
        'frames': int(summary['measured_frames']),
        'parameters_m': float(summary['parameter_count_m']),
        'decode_mean_ms': float(latency['decode_ms']['mean']),
        'preprocess_mean_ms': float(latency['preprocess_ms']['mean']),
        'model_mean_ms': float(latency['model_ms']['mean']),
        'filter_mean_ms': float(latency['filter_ms']['mean']),
        'pipeline_mean_ms': float(latency['pipeline_ms']['mean']),
        'pipeline_p95_ms': float(latency['pipeline_ms']['p95']),
        'total_mean_ms': float(latency['total_no_render_ms']['mean']),
        'total_median_ms': float(latency['total_no_render_ms']['median']),
        'total_p95_ms': float(latency['total_no_render_ms']['p95']),
        'fps_e2e': float(summary['fps_end_to_end_no_render']),
        'peak_memory_mb': float(summary['gpu_peak_memory_mb']),
        'peak_reserved_memory_mb': float(
            summary.get('gpu_peak_reserved_memory_mb', 0.0)),
    }


def aggregate(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)
    numeric = [
        'frames', 'parameters_m', 'decode_mean_ms', 'preprocess_mean_ms',
        'model_mean_ms', 'filter_mean_ms', 'pipeline_mean_ms',
        'pipeline_p95_ms', 'total_mean_ms', 'total_median_ms',
        'total_p95_ms', 'fps_e2e', 'peak_memory_mb',
        'peak_reserved_memory_mb'
    ]
    output = []
    for key, items in sorted(grouped.items()):
        result = dict(zip(group_keys, key))
        result['trials'] = len(items)
        for name in numeric:
            result[name] = median([float(item[name]) for item in items])
        result['total_mean_ms_min'] = min(
            float(item['total_mean_ms']) for item in items)
        result['total_mean_ms_max'] = max(
            float(item['total_mean_ms']) for item in items)
        output.append(result)
    return output


def collect_e5() -> tuple[list[dict], list[dict]]:
    accuracy = {
        size: latest_metric_record(
            STAGE1_DIR / f'E5_accuracy_s{size}_fp32')
        for size in (512, 640, 768)
    }
    trials = []
    for size in (512, 640, 768):
        for trial in (1, 2, 3):
            summary = load_json(
                STAGE1_DIR
                / f'E5_latency_s{size}_fp32_trial{trial}'
                / 'summary.json')
            trials.append(trial_row(summary, trial, 'RemDet-S'))
    summary_rows = aggregate(trials, ('model', 'input_size', 'precision'))
    for row in summary_rows:
        row.update(accuracy[int(row['input_size'])])
    return trials, summary_rows


def collect_e6() -> tuple[list[dict], list[dict]]:
    fp32_accuracy = {
        'Tiny-640': latest_metric_record(
            PROJECT_ROOT / 'work_dirs' / 'video_experiments' / 'E0_accuracy'),
        'S-640': latest_metric_record(
            STAGE1_DIR / 'E5_accuracy_s640_fp32'),
        'S-768': latest_metric_record(
            STAGE1_DIR / 'E5_accuracy_s768_fp32'),
    }
    amp_accuracy = {
        'Tiny-640': latest_metric_record(
            STAGE1_DIR / 'E6_accuracy_tiny640_amp'),
        'S-640': latest_metric_record(
            STAGE1_DIR / 'E6_accuracy_s640_amp'),
        'S-768': latest_metric_record(
            STAGE1_DIR / 'E6_accuracy_s768_amp'),
    }
    definitions = [
        ('tiny', 640, 'Tiny-640'),
        ('s', 640, 'S-640'),
        ('s', 768, 'S-768'),
    ]
    trials = []
    for tag, size, label in definitions:
        for precision, directory_tag in (
                ('fp32', 'fp32'), ('amp-fp16', 'amp_fp16')):
            for trial in (1, 2, 3):
                path = (
                    STAGE1_DIR
                    / f'E6_latency_{tag}{size}_{directory_tag}_trial{trial}'
                    / 'summary.json')
                trials.append(trial_row(load_json(path), trial, label))
    summary_rows = aggregate(trials, ('model', 'input_size', 'precision'))
    for row in summary_rows:
        metrics = (
            amp_accuracy[row['model']]
            if row['precision'] == 'amp-fp16'
            else fp32_accuracy[row['model']])
        row.update(metrics)

    by_model = {(row['model'], row['precision']): row for row in summary_rows}
    for row in summary_rows:
        base = by_model[(row['model'], 'fp32')]
        row['map_delta_vs_fp32'] = row['map'] - base['map']
        row['latency_change_pct_vs_fp32'] = (
            row['total_mean_ms'] / base['total_mean_ms'] - 1.0) * 100.0
        row['speedup_vs_fp32'] = (
            base['total_mean_ms'] / row['total_mean_ms'])
        row['memory_change_pct_vs_fp32'] = (
            row['peak_memory_mb'] / base['peak_memory_mb'] - 1.0) * 100.0
    return trials, summary_rows


def setup_plot_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 180,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.22,
        'font.size': 9,
    })


def plot_e5(summary: list[dict]) -> None:
    rows = sorted(summary, key=lambda row: row['input_size'])
    sizes = [row['input_size'] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(sizes, [row['map'] * 100 for row in rows], 'o-',
                 lw=2.2, label='mAP50:95')
    axes[0].plot(sizes, [row['aps'] * 100 for row in rows], 's--',
                 lw=2.0, label='AP small')
    axes[0].set(title='E5 Accuracy vs. resolution', xlabel='Input size',
                ylabel='AP (%)', xticks=sizes)
    axes[0].legend(frameon=False)

    axes[1].plot(sizes, [row['total_mean_ms'] for row in rows], 'o-',
                 lw=2.2, color='#d97706', label='Latency')
    axes[1].set(title='E5 End-to-end latency', xlabel='Input size',
                ylabel='Latency (ms/frame)', xticks=sizes)
    second = axes[1].twinx()
    second.plot(sizes, [row['fps_e2e'] for row in rows], 's--',
                lw=2.0, color='#0f766e', label='FPS')
    second.set_ylabel('FPS')
    lines = axes[1].lines + second.lines
    axes[1].legend(lines, [line.get_label() for line in lines],
                   frameon=False, loc='best')
    fig.suptitle('RemDet-S resolution trade-off (median of 3 trials)')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'e5_resolution_tradeoff.png', bbox_inches='tight')
    plt.close(fig)

    labels = [str(size) for size in sizes]
    components = [
        ('Decode', 'decode_mean_ms', '#94a3b8'),
        ('Preprocess', 'preprocess_mean_ms', '#60a5fa'),
        ('Model + NMS', 'model_mean_ms', '#2563eb'),
        ('Filter', 'filter_mean_ms', '#14b8a6'),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bottom = np.zeros(len(rows))
    for label, key, color in components:
        values = np.array([row[key] for row in rows])
        ax.bar(labels, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set(title='E5 latency breakdown', xlabel='Input size',
           ylabel='Mean ms/frame')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'e5_latency_breakdown.png', bbox_inches='tight')
    plt.close(fig)


def plot_e6(summary: list[dict]) -> None:
    model_order = ['Tiny-640', 'S-640', 'S-768']
    by_key = {(row['model'], row['precision']): row for row in summary}
    x = np.arange(len(model_order))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for index, (metric, title) in enumerate(
            (('map', 'mAP50:95'), ('aps', 'AP small'))):
        fp = [by_key[(model, 'fp32')][metric] * 100 for model in model_order]
        amp = [by_key[(model, 'amp-fp16')][metric] * 100
               for model in model_order]
        axes[index].bar(x - width / 2, fp, width, label='FP32',
                        color='#475569')
        axes[index].bar(x + width / 2, amp, width, label='AMP-FP16',
                        color='#0ea5e9')
        axes[index].set(title=f'E6 {title}', ylabel='AP (%)',
                        xticks=x, xticklabels=model_order)
        axes[index].legend(frameon=False)
    fig.suptitle('Accuracy is unchanged at reported COCO precision')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'e6_accuracy_comparison.png',
                bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    fp_latency = [by_key[(model, 'fp32')]['total_mean_ms']
                  for model in model_order]
    amp_latency = [by_key[(model, 'amp-fp16')]['total_mean_ms']
                   for model in model_order]
    axes[0].bar(x - width / 2, fp_latency, width, label='FP32',
                color='#475569')
    axes[0].bar(x + width / 2, amp_latency, width, label='AMP-FP16',
                color='#0ea5e9')
    axes[0].set(title='End-to-end latency', ylabel='ms/frame',
                xticks=x, xticklabels=model_order)
    axes[0].legend(frameon=False)

    fp_memory = [by_key[(model, 'fp32')]['peak_memory_mb']
                 for model in model_order]
    amp_memory = [by_key[(model, 'amp-fp16')]['peak_memory_mb']
                  for model in model_order]
    axes[1].bar(x - width / 2, fp_memory, width, label='FP32',
                color='#475569')
    axes[1].bar(x + width / 2, amp_memory, width, label='AMP-FP16',
                color='#0ea5e9')
    axes[1].set(title='Peak allocated GPU memory', ylabel='MiB',
                xticks=x, xticklabels=model_order)
    axes[1].legend(frameon=False)
    fig.suptitle('E6 AMP deployment trade-off (median of 3 trials)')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'e6_latency_memory.png', bbox_inches='tight')
    plt.close(fig)


def format_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    header = '| ' + ' | '.join(label for _, label, _ in columns) + ' |'
    divider = '| ' + ' | '.join('---' for _ in columns) + ' |'
    body = []
    for row in rows:
        cells = []
        for key, _, fmt in columns:
            value = row[key]
            cells.append(format(value, fmt) if fmt else str(value))
        body.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join([header, divider, *body])


def build_report(e5: list[dict], e6: list[dict]) -> None:
    e5_rows = sorted(e5, key=lambda row: row['input_size'])
    e6_rows = sorted(
        e6,
        key=lambda row: (
            ['Tiny-640', 'S-640', 'S-768'].index(row['model']),
            0 if row['precision'] == 'fp32' else 1))
    e5_table = format_table(e5_rows, [
        ('input_size', '输入', 'd'), ('map', 'mAP', '.3f'),
        ('ap50', 'AP50', '.3f'), ('aps', 'AP-small', '.3f'),
        ('total_mean_ms', '端到端均值(ms)', '.3f'),
        ('total_p95_ms', 'P95(ms)', '.3f'), ('fps_e2e', 'FPS', '.1f'),
        ('peak_memory_mb', '峰值显存(MiB)', '.1f'),
    ])
    e6_table = format_table(e6_rows, [
        ('model', '组合', ''), ('precision', '精度模式', ''),
        ('map', 'mAP', '.3f'), ('aps', 'AP-small', '.3f'),
        ('total_mean_ms', '端到端均值(ms)', '.3f'),
        ('total_p95_ms', 'P95(ms)', '.3f'), ('fps_e2e', 'FPS', '.1f'),
        ('peak_memory_mb', '峰值显存(MiB)', '.1f'),
        ('latency_change_pct_vs_fp32', '延迟变化', '+.1f'),
        ('memory_change_pct_vs_fp32', '显存变化', '+.1f'),
    ])

    e5_512, e5_640, e5_768 = e5_rows
    by_key = {(row['model'], row['precision']): row for row in e6_rows}
    conclusions = []
    for model in ('Tiny-640', 'S-640', 'S-768'):
        amp = by_key[(model, 'amp-fp16')]
        conclusions.append(
            f"- {model}：AMP 延迟变化 {amp['latency_change_pct_vs_fp32']:+.1f}% "
            f"，峰值显存变化 {amp['memory_change_pct_vs_fp32']:+.1f}% 。")

    report = f"""# 第一阶段 E5/E6 实验报告

生成时间：2026-08-26。硬件：NVIDIA GeForce RTX 5080；PyTorch 2.7.1+cu128；CUDA 12.8。

## 实验设置

| 实验 | 变量 | 固定条件 | 数据量 |
| --- | --- | --- | --- |
| E5 | RemDet-S 输入 512 / 640 / 768 | 官方 S 权重、FP32、同一 VisDrone val | 精度：548 张；性能：每档 3×670 帧 |
| E6 | FP32 / AMP-FP16 | Tiny-640、S-640、S-768；同一权重和输入 | 精度：每组合 548 张；性能：每组合 3×670 帧 |

性能表采用 3 个独立试次的中位数；每次先预热 50 帧，再对 67 帧视频重复 10 次。端到端时间包含视频解码、预处理、模型前向、框解码/NMS和结果过滤，不含绘制与写视频。

## E5：RemDet-S 多分辨率

{e5_table}

结果：512→768 的 mAP 提升 {(e5_768['map'] - e5_512['map']) * 100:.1f} 个百分点，小目标 AP 提升 {(e5_768['aps'] - e5_512['aps']) * 100:.1f} 个百分点。640→768 仍有 mAP +{(e5_768['map'] - e5_640['map']) * 100:.1f}、AP-small +{(e5_768['aps'] - e5_640['aps']) * 100:.1f} 个百分点。S-640 在本机上还比 S-512 略快，因此 640 是当前效率基线；768 用约 {(e5_768['total_mean_ms'] / e5_640['total_mean_ms'] - 1) * 100:.1f}% 端到端延迟换取明显的小目标收益，适合作为精度基线。512 只有峰值显存更低，不能作为“更快”的方案。

![E5 trade-off](e5_resolution_tradeoff.png)

![E5 breakdown](e5_latency_breakdown.png)

## E6：FP32 与 AMP-FP16

{e6_table}

COCO 指标保留 3 位小数时，所有 AMP 组合与 FP32 完全一致，说明当前混合精度没有可见精度损失。性能结论必须以实测为准：

{chr(10).join(conclusions)}

本环境的 AMP 只把卷积等网络计算放进 FP16；模型参数仍以 FP32 保存，本地 MMCV NMS 也必须转回 FP32。因而它是“自动混合精度推理”，不是 TensorRT 那类完整 FP16 部署图。

![E6 accuracy](e6_accuracy_comparison.png)

![E6 latency and memory](e6_latency_memory.png)

## 阶段结论

1. 当前最值得保留的科研变量是输入分辨率：**S-640 FP32** 是效率基线，**S-768 FP32** 是高精度/小目标基线。
2. 当前 AMP 在精度上安全、在显存上有效，但单帧端到端延迟更高，因此不作为默认实时方案；只有显存受限时才优先考虑。
3. S-512 在 RTX 5080 上没有速度优势，后续不必作为主线；等网络结构成熟后再进入你暂缓的第二阶段实验。

## 文件说明

- `e5_resolution_trials.csv` / `e5_resolution_summary.csv`：E5 原始试次与汇总。
- `e6_precision_trials.csv` / `e6_precision_summary.csv`：E6 原始试次与汇总。
- 所有 `summary.json`、逐帧 `latency_samples.json` 和验证日志保留在 `work_dirs/video_experiments/stage1/`。
"""
    (OUTPUT_DIR / 'stage1_report.md').write_text(report, encoding='utf-8')


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    e5_trials, e5_summary = collect_e5()
    e6_trials, e6_summary = collect_e6()
    write_csv(OUTPUT_DIR / 'e5_resolution_trials.csv', e5_trials)
    write_csv(OUTPUT_DIR / 'e5_resolution_summary.csv', e5_summary)
    write_csv(OUTPUT_DIR / 'e6_precision_trials.csv', e6_trials)
    write_csv(OUTPUT_DIR / 'e6_precision_summary.csv', e6_summary)
    plot_e5(e5_summary)
    plot_e6(e6_summary)
    build_report(e5_summary, e6_summary)
    print(f'Report written to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
