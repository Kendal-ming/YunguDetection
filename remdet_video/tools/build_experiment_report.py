"""Aggregate E0-E4 results and create advisor-ready tables and plots."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / 'work_dirs' / 'video_experiments'
REPORT_ROOT = EXPERIMENT_ROOT / 'report'


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_accuracy(directory: Path) -> dict[str, float]:
    logs = sorted(directory.rglob('*.log'), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f'No test log found under {directory}')
    pattern = re.compile(
        r'coco/bbox_mAP: ([0-9.]+).*?'
        r'coco/bbox_mAP_50: ([0-9.]+).*?'
        r'coco/bbox_mAP_75: ([0-9.]+).*?'
        r'coco/bbox_mAP_s: ([0-9.]+).*?'
        r'coco/bbox_mAP_m: ([0-9.]+).*?'
        r'coco/bbox_mAP_l: ([0-9.]+)')
    matches = pattern.findall(logs[-1].read_text(encoding='utf-8'))
    if not matches:
        raise ValueError(f'No COCO metric line found in {logs[-1]}')
    values = map(float, matches[-1])
    keys = ('mAP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large')
    return dict(zip(keys, values))


def median(values):
    return statistics.median(float(value) for value in values)


def build_resolution_tables() -> tuple[list[dict], list[dict]]:
    trial_rows = []
    grouped = defaultdict(list)
    for size in (512, 640, 768):
        for trial in (1, 2, 3):
            summary = read_json(
                EXPERIMENT_ROOT / f'E2_{size}_trial{trial}' / 'summary.json')
            latency = summary['latency']
            row = {
                'input_size': size,
                'trial': trial,
                'measured_frames': summary['measured_frames'],
                'decode_mean_ms': latency['decode_ms']['mean'],
                'preprocess_mean_ms': latency['preprocess_ms']['mean'],
                'model_wall_mean_ms': latency['model_ms']['mean'],
                'model_gpu_mean_ms': latency['gpu_model_ms']['mean'],
                'filter_mean_ms': latency['filter_ms']['mean'],
                'pipeline_mean_ms': latency['pipeline_ms']['mean'],
                'total_mean_ms': latency['total_no_render_ms']['mean'],
                'total_p95_ms': latency['total_no_render_ms']['p95'],
                'fps_core': summary['fps_core'],
                'fps_end_to_end': summary['fps_end_to_end_no_render'],
                'gpu_peak_memory_mb': summary['gpu_peak_memory_mb'],
            }
            trial_rows.append(row)
            grouped[size].append(row)

    accuracy = {
        512: parse_accuracy(EXPERIMENT_ROOT / 'E2_accuracy_512'),
        640: parse_accuracy(EXPERIMENT_ROOT / 'E0_accuracy'),
        768: parse_accuracy(EXPERIMENT_ROOT / 'E2_accuracy_768'),
    }
    summary_rows = []
    for size, rows in grouped.items():
        total_means = [row['total_mean_ms'] for row in rows]
        summary_rows.append({
            'input_size': size,
            'trials': len(rows),
            'measured_frames': sum(row['measured_frames'] for row in rows),
            'mAP': accuracy[size]['mAP'],
            'AP50': accuracy[size]['AP50'],
            'AP75': accuracy[size]['AP75'],
            'AP_small': accuracy[size]['AP_small'],
            'AP_medium': accuracy[size]['AP_medium'],
            'AP_large': accuracy[size]['AP_large'],
            'decode_mean_ms_median': median(
                row['decode_mean_ms'] for row in rows),
            'preprocess_mean_ms_median': median(
                row['preprocess_mean_ms'] for row in rows),
            'model_wall_mean_ms_median': median(
                row['model_wall_mean_ms'] for row in rows),
            'model_gpu_mean_ms_median': median(
                row['model_gpu_mean_ms'] for row in rows),
            'filter_mean_ms_median': median(
                row['filter_mean_ms'] for row in rows),
            'pipeline_mean_ms_median': median(
                row['pipeline_mean_ms'] for row in rows),
            'total_mean_ms_median': median(total_means),
            'total_mean_ms_trial_std': statistics.stdev(total_means),
            'total_p95_ms_median': median(
                row['total_p95_ms'] for row in rows),
            'fps_core_median': median(row['fps_core'] for row in rows),
            'fps_end_to_end_median': median(
                row['fps_end_to_end'] for row in rows),
            'gpu_peak_memory_mb_median': median(
                row['gpu_peak_memory_mb'] for row in rows),
        })
    return trial_rows, summary_rows


def build_model_tables() -> tuple[list[dict], list[dict]]:
    definitions = {
        'Tiny': [
            'E2_640_trial1', 'E2_640_trial2', 'E2_640_trial3',
            'E1_tiny_640_trial4', 'E1_tiny_640_trial5',
            'E1_tiny_640_trial6',
        ],
        'S': [f'E1_s_640_trial{trial}' for trial in range(1, 7)],
    }
    accuracy = {
        'Tiny': parse_accuracy(EXPERIMENT_ROOT / 'E0_accuracy'),
        'S': parse_accuracy(EXPERIMENT_ROOT / 'E1_accuracy_s_safe'),
    }
    profiles = {
        'Tiny': read_json(
            EXPERIMENT_ROOT / 'E1_tiny_stage_profile' / 'summary.json'),
        'S': read_json(
            EXPERIMENT_ROOT / 'E1_s_stage_profile' / 'summary.json'),
    }
    trial_rows = []
    summary_rows = []
    for model_name, directories in definitions.items():
        model_trials = []
        for trial, directory in enumerate(directories, start=1):
            summary = read_json(EXPERIMENT_ROOT / directory / 'summary.json')
            latency = summary['latency']
            row = {
                'model': model_name,
                'trial': trial,
                'directory': directory,
                'measured_frames': summary['measured_frames'],
                'parameter_count_m_training_graph': summary['parameter_count_m'],
                'total_mean_ms': latency['total_no_render_ms']['mean'],
                'total_p95_ms': latency['total_no_render_ms']['p95'],
                'pipeline_mean_ms': latency['pipeline_ms']['mean'],
                'gpu_model_mean_ms': latency['gpu_model_ms']['mean'],
                'fps_core': summary['fps_core'],
                'fps_end_to_end': summary['fps_end_to_end_no_render'],
                'gpu_peak_memory_mb': summary['gpu_peak_memory_mb'],
            }
            trial_rows.append(row)
            model_trials.append(row)
        profile = profiles[model_name]
        summary_rows.append({
            'model': model_name,
            'trials': len(model_trials),
            'measured_frames': sum(
                row['measured_frames'] for row in model_trials),
            'mAP': accuracy[model_name]['mAP'],
            'AP50': accuracy[model_name]['AP50'],
            'AP75': accuracy[model_name]['AP75'],
            'AP_small': accuracy[model_name]['AP_small'],
            'AP_medium': accuracy[model_name]['AP_medium'],
            'AP_large': accuracy[model_name]['AP_large'],
            'parameter_count_m_training_graph': median(
                row['parameter_count_m_training_graph']
                for row in model_trials),
            'total_mean_ms_median': median(
                row['total_mean_ms'] for row in model_trials),
            'total_p95_ms_median': median(
                row['total_p95_ms'] for row in model_trials),
            'fps_core_median': median(
                row['fps_core'] for row in model_trials),
            'fps_end_to_end_median': median(
                row['fps_end_to_end'] for row in model_trials),
            'gpu_peak_memory_mb_median': median(
                row['gpu_peak_memory_mb'] for row in model_trials),
            'raw_gpu_total_ms': profile['timing']['raw_gpu_total_ms']['mean'],
            'backbone_neck_gpu_ms': (
                profile['timing']['backbone_neck_gpu_ms']['mean']),
            'head_gpu_ms': profile['timing']['head_gpu_ms']['mean'],
            'decode_nms_gpu_ms': (
                profile['timing']['decode_nms_gpu_ms']['mean']),
        })
    return trial_rows, summary_rows


def configure_plot() -> None:
    plt.rcParams.update({
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })


def plot_resolution_tradeoff(rows: list[dict]) -> None:
    configure_plot()
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    x = [row['total_mean_ms_median'] for row in rows]
    y = [row['mAP'] * 100 for row in rows]
    axis.plot(x, y, color='#3178c6', linewidth=1.5)
    axis.scatter(x, y, s=85, color='#3178c6', zorder=3)
    for row, x_value, y_value in zip(rows, x, y):
        axis.annotate(
            f"{row['input_size']} px\n{y_value:.1f} mAP, {x_value:.2f} ms",
            (x_value, y_value), xytext=(7, 7), textcoords='offset points')
    axis.set_xlabel('End-to-end latency without rendering (ms/frame)')
    axis.set_ylabel('VisDrone bbox mAP (%)')
    axis.set_title('Resolution: accuracy–latency trade-off (RTX 5080)')
    axis.grid(axis='both', alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / 'resolution_tradeoff.png', bbox_inches='tight')
    plt.close(fig)


def plot_latency_breakdown(rows: list[dict]) -> None:
    configure_plot()
    labels = [str(row['input_size']) for row in rows]
    components = {
        'Decode': [row['decode_mean_ms_median'] for row in rows],
        'Preprocess': [row['preprocess_mean_ms_median'] for row in rows],
        'Model + decode/NMS': [row['model_wall_mean_ms_median'] for row in rows],
        'Result filter': [row['filter_mean_ms_median'] for row in rows],
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    bottoms = np.zeros(len(rows))
    colors = ('#5b8ff9', '#61d9a4', '#f6bd16', '#e8684a')
    for (name, values), color in zip(components.items(), colors):
        axis.bar(labels, values, bottom=bottoms, label=name, color=color)
        bottoms += np.asarray(values)
    for index, total in enumerate(bottoms):
        axis.text(index, total + 0.15, f'{total:.2f} ms', ha='center')
    axis.set_xlabel('Square test input (pixels)')
    axis.set_ylabel('Mean wall latency (ms/frame)')
    axis.set_title('Per-frame latency breakdown (median of 3 trials)')
    axis.legend(frameon=False, ncol=2)
    axis.set_ylim(0, max(bottoms) * 1.18)
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / 'latency_breakdown.png', bbox_inches='tight')
    plt.close(fig)


def plot_model_comparison(rows: list[dict]) -> None:
    configure_plot()
    labels = [row['model'] for row in rows]
    map_values = [row['mAP'] * 100 for row in rows]
    latency_values = [row['total_mean_ms_median'] for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.4))
    axes[0].bar(x, map_values, color=('#5b8ff9', '#61d9a4'), width=0.58)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel('VisDrone bbox mAP (%)')
    axes[0].set_title('Accuracy')
    axes[0].set_ylim(0, max(map_values) * 1.25)
    for index, value in enumerate(map_values):
        axes[0].text(index, value + 0.4, f'{value:.1f}', ha='center')
    axes[1].bar(
        x, latency_values, color=('#5b8ff9', '#61d9a4'), width=0.58)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel('End-to-end latency (ms/frame)')
    axes[1].set_title('Video latency, no rendering')
    axes[1].set_ylim(0, max(latency_values) * 1.25)
    for index, value in enumerate(latency_values):
        axes[1].text(index, value + 0.18, f'{value:.2f}', ha='center')
    fig.suptitle('RemDet-Tiny vs RemDet-S at 640 px (RTX 5080)')
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / 'model_comparison.png', bbox_inches='tight')
    plt.close(fig)


def plot_threshold_tradeoff(rows: list[dict]) -> None:
    configure_plot()
    selected = sorted(
        (row for row in rows
         if row['scope'] == 'micro' and float(row['iou_threshold']) == 0.5),
        key=lambda row: float(row['score_threshold']))
    thresholds = [float(row['score_threshold']) for row in selected]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for key, label, color in (
            ('precision', 'Precision', '#3178c6'),
            ('recall', 'Recall', '#e8684a'),
            ('f1', 'F1', '#36a269')):
        values = [float(row[key]) * 100 for row in selected]
        axis.plot(thresholds, values, marker='o', label=label, color=color)
        if key == 'f1':
            best = max(range(len(values)), key=values.__getitem__)
            axis.annotate(
                f'best F1={values[best]:.1f}% at {thresholds[best]:.1f}',
                (thresholds[best], values[best]), xytext=(8, 10),
                textcoords='offset points')
    axis.set_xlabel('Score threshold')
    axis.set_ylabel('Metric at IoU=0.50 (%)')
    axis.set_title('Operational threshold trade-off on VisDrone val')
    axis.set_xticks(thresholds)
    axis.grid(axis='both', alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / 'threshold_tradeoff.png', bbox_inches='tight')
    plt.close(fig)


def plot_threshold_f1_models(
        tiny_rows: list[dict], s_rows: list[dict]) -> None:
    configure_plot()
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for name, rows, color in (
            ('Tiny', tiny_rows, '#3178c6'), ('S', s_rows, '#36a269')):
        selected = sorted(
            (row for row in rows
             if row['scope'] == 'micro'
             and float(row['iou_threshold']) == 0.5),
            key=lambda row: float(row['score_threshold']))
        thresholds = [float(row['score_threshold']) for row in selected]
        values = [float(row['f1']) * 100 for row in selected]
        axis.plot(thresholds, values, marker='o', label=name, color=color)
        best = max(range(len(values)), key=values.__getitem__)
        axis.annotate(
            f'{name}: {values[best]:.1f}% @ {thresholds[best]:.1f}',
            (thresholds[best], values[best]), xytext=(7, 8),
            textcoords='offset points')
    axis.set_xlabel('Score threshold')
    axis.set_ylabel('Micro-F1 at IoU=0.50 (%)')
    axis.set_title('Threshold sensitivity by model')
    axis.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    axis.grid(axis='both', alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / 'threshold_model_f1.png', bbox_inches='tight')
    plt.close(fig)


def plot_temporal_proxy(rows: list[dict]) -> None:
    configure_plot()
    selected = [
        row for row in rows
        if row['target'] == 'bus' and float(row['threshold']) == 0.3]
    rule_order = {
        'T0_1of1_exit1': 0,
        'T1_2of3_exit2': 1,
        'T2_3of5_exit3': 2,
        'T3_2of5_exit5': 3,
    }
    selected.sort(key=lambda row: rule_order[row['rule']])
    labels = ['T0', 'T1', 'T2', 'T3']
    transitions = [int(row['filtered_transitions']) for row in selected]
    delay = [float(row['mean_proxy_entry_delay_frames']) for row in selected]
    x = np.arange(len(labels))
    fig, first = plt.subplots(figsize=(7.2, 4.6))
    first.bar(x, transitions, color='#5b8ff9', width=0.55,
              label='Filtered transitions')
    first.set_ylabel('Detection-state transitions (count)')
    first.set_xlabel('Temporal rule')
    first.set_xticks(x, labels)
    second = first.twinx()
    second.plot(x, delay, color='#e8684a', marker='o', linewidth=2,
                label='Proxy entry delay')
    second.set_ylabel('Proxy entry delay (frames)')
    first.set_title('Temporal smoothing example: bus, threshold 0.30')
    handles_a, labels_a = first.get_legend_handles_labels()
    handles_b, labels_b = second.get_legend_handles_labels()
    first.legend(handles_a + handles_b, labels_a + labels_b,
                 frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / 'temporal_proxy.png', bbox_inches='tight')
    plt.close(fig)


def fmt(value: float, decimals: int = 3) -> str:
    return f'{float(value):.{decimals}f}'


def build_markdown(
        model_rows: list[dict], resolution_rows: list[dict],
        threshold_rows: list[dict], threshold_rows_s: list[dict],
        temporal_rows: list[dict]) -> str:
    baseline = read_json(
        EXPERIMENT_ROOT / 'E0_tiny_640_fp32' / 'summary.json')
    output = read_json(
        EXPERIMENT_ROOT / 'E0_tiny_640_fp32_with_output' / 'summary.json')
    micro = sorted(
        (row for row in threshold_rows
         if row['scope'] == 'micro' and float(row['iou_threshold']) == 0.5),
        key=lambda row: float(row['score_threshold']))
    best_f1 = max(micro, key=lambda row: float(row['f1']))
    micro_s = sorted(
        (row for row in threshold_rows_s
         if row['scope'] == 'micro' and float(row['iou_threshold']) == 0.5),
        key=lambda row: float(row['score_threshold']))
    best_f1_s = max(micro_s, key=lambda row: float(row['f1']))
    tiny_model = next(row for row in model_rows if row['model'] == 'Tiny')
    s_model = next(row for row in model_rows if row['model'] == 'S')
    bus = [
        row for row in temporal_rows
        if row['target'] == 'bus' and float(row['threshold']) == 0.3]
    t1 = next(row for row in bus if row['rule'].startswith('T1_'))

    lines = [
        '# RemDet 视频目标检测 E0–E4 阶段实验报告',
        '',
        '生成日期：2026-08-25。硬件：NVIDIA GeForce RTX 5080；软件：'
        f"PyTorch {baseline['environment']['torch']} / CUDA "
        f"{baseline['environment']['torch_cuda']}。",
        '',
        '## 核心结论',
        '',
        f"- E0：Tiny 640 的 VisDrone bbox mAP 为 0.213，AP50 为 0.363。"
        f"长测核心吞吐为 {baseline['fps_core']:.1f} FPS；不画框端到端 "
        f"{baseline['fps_end_to_end_no_render']:.1f} FPS。",
        f"- 完整输出链路（解码、推理、画框、编码）均值 "
        f"{output['latency']['total_with_output_ms']['mean']:.2f} ms/帧，"
        f"约 {1000 / output['latency']['total_with_output_ms']['mean']:.1f} FPS。",
        f"- E1：S 的 mAP 为 {s_model['mAP']:.3f}，比 Tiny 高 "
        f"{s_model['mAP'] - tiny_model['mAP']:.3f}；在 RTX 5080 上，S 的"
        f"端到端中位延迟反而由 {tiny_model['total_mean_ms_median']:.2f} "
        f"降至 {s_model['total_mean_ms_median']:.2f} ms。该速度排序不能直接"
        '外推到机器人端设备。',
        '- E2：512 没有带来稳定加速，却使 mAP 比 640 低 0.030；'
        '768 使 mAP 提高 0.021，但端到端延迟增加约 1 ms。',
        f"- E3：IoU=0.50 下，当前五个候选阈值中 {best_f1['score_threshold']} "
        f"的 micro-F1 最高（{float(best_f1['f1']):.3f}）。",
        f"- E4：在无事件标注 demo 上，T1 把 bus 状态跳变由 18 次降为 "
        f"{t1['filtered_transitions']} 次，代理进入延迟为 "
        f"{float(t1['mean_proxy_entry_delay_frames']):.3f} 帧；"
        '这只是稳定性代理结果，不能当成真实事件精确率或召回率。',
        '- 代码审计：仓库的 `RepDWConv.switch_to_deploy()` 未通过等价性检查，'
        '会因分组卷积权重形状错误而崩溃；当前表格明确使用可运行的训练图推理。',
        '',
        '## E0 基线',
        '',
        '| 配置 | 验证图 | mAP | AP50 | AP75 | APs | APm | APl |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        '| Tiny 640 FP32 | 548 | 0.213 | 0.363 | 0.214 | 0.119 | 0.325 | 0.438 |',
        '',
        '| 视频计时口径 | 帧数 | 均值 ms | P95 ms | FPS |',
        '|---|---:|---:|---:|---:|',
        f"| 核心流水线（预处理+模型+结果过滤） | {baseline['measured_frames']} | "
        f"{baseline['latency']['pipeline_ms']['mean']:.3f} | "
        f"{baseline['latency']['pipeline_ms']['p95']:.3f} | "
        f"{baseline['fps_core']:.1f} |",
        f"| 端到端、不画框 | {baseline['measured_frames']} | "
        f"{baseline['latency']['total_no_render_ms']['mean']:.3f} | "
        f"{baseline['latency']['total_no_render_ms']['p95']:.3f} | "
        f"{baseline['fps_end_to_end_no_render']:.1f} |",
        f"| 端到端、画框并保存 | {output['measured_frames']} | "
        f"{output['latency']['total_with_output_ms']['mean']:.3f} | "
        f"{output['latency']['total_with_output_ms']['p95']:.3f} | "
        f"{1000 / output['latency']['total_with_output_ms']['mean']:.1f} |",
        '',
        '## E1 Tiny 与 S',
        '',
        '每个模型独立运行 6 轮，每轮 670 帧；速度为六轮中位数。参数量是当前'
        '可运行训练图的实测值，不与论文部署图参数混写。',
        '',
        '| 模型 | 累计计时帧 | mAP | AP50 | APs | 参数 M | 均值延迟 ms | P95 ms | 端到端 FPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {row['measured_frames']} | "
            f"{fmt(row['mAP'])} | {fmt(row['AP50'])} | "
            f"{fmt(row['AP_small'])} | "
            f"{fmt(row['parameter_count_m_training_graph'], 2)} | "
            f"{fmt(row['total_mean_ms_median'])} | "
            f"{fmt(row['total_p95_ms_median'])} | "
            f"{fmt(row['fps_end_to_end_median'], 1)} |")
    lines.extend([
        '',
        '![模型对比](model_comparison.png)',
        '',
        '| 模型 | 主干+颈部 GPU ms | 检测头 GPU ms | 解码+NMS GPU ms | GPU 合计 ms |',
        '|---|---:|---:|---:|---:|',
    ])
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {fmt(row['backbone_neck_gpu_ms'])} | "
            f"{fmt(row['head_gpu_ms'])} | {fmt(row['decode_nms_gpu_ms'])} | "
            f"{fmt(row['raw_gpu_total_ms'])} |")
    lines.extend([
        '',
        '## E2 分辨率实验',
        '',
        '每个分辨率独立运行 3 轮，每轮 670 帧；表内延迟和 FPS 为三轮中位数。',
        '',
        '| 输入 | 累计计时帧 | mAP | AP50 | APs | 均值延迟 ms | P95 ms | 核心 FPS | 端到端 FPS |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in resolution_rows:
        lines.append(
            f"| {row['input_size']} | {row['measured_frames']} | "
            f"{fmt(row['mAP'])} | {fmt(row['AP50'])} | {fmt(row['AP_small'])} | "
            f"{fmt(row['total_mean_ms_median'])} | "
            f"{fmt(row['total_p95_ms_median'])} | "
            f"{fmt(row['fps_core_median'], 1)} | "
            f"{fmt(row['fps_end_to_end_median'], 1)} |")
    lines.extend([
        '',
        '![分辨率速度精度权衡](resolution_tradeoff.png)',
        '',
        '![分辨率延迟分解](latency_breakdown.png)',
        '',
        '## E3 置信度阈值（固定 IoU=0.50）',
        '',
        'Tiny：',
        '',
        '| 阈值 | Precision | Recall | F1 | 每图检测数 |',
        '|---:|---:|---:|---:|---:|',
    ])
    for row in micro:
        lines.append(
            f"| {float(row['score_threshold']):.2f} | "
            f"{fmt(row['precision'])} | {fmt(row['recall'])} | "
            f"{fmt(row['f1'])} | {float(row['detections_per_image']):.1f} |")
    lines.extend([
        '',
        '![阈值权衡](threshold_tradeoff.png)',
        '',
        'S：',
        '',
        '| 阈值 | Precision | Recall | F1 | 每图检测数 |',
        '|---:|---:|---:|---:|---:|',
    ])
    for row in micro_s:
        lines.append(
            f"| {float(row['score_threshold']):.2f} | "
            f"{fmt(row['precision'])} | {fmt(row['recall'])} | "
            f"{fmt(row['f1'])} | {float(row['detections_per_image']):.1f} |")
    lines.extend([
        '',
        '![模型阈值F1对比](threshold_model_f1.png)',
        '',
        f"当前候选中，Tiny 最佳 F1 阈值为 {best_f1['score_threshold']}；"
        f"S 最佳为 {best_f1_s['score_threshold']}。",
        '',
        '说明：这是固定分数阈值下的逐框匹配指标，用于工程选阈值；'
        '它与 COCO 的 101 点插值 mAP 不是同一个统计量。',
        '',
        '## E4 时间规则（无真值标签的代理实验）',
        '',
        'demo.mp4 只有 67 帧且车辆几乎全程存在，因此本轮只证明时间规则可运行、'
        '开销很小并能抑制状态闪烁。正式论文结论仍需要带“目标进入/离开”时间标注的视频。',
        '',
        '![时间平滑代理结果](temporal_proxy.png)',
        '',
        '在当前 RTX 5080 上，建议下一阶段以 `S + 640 + 阈值0.30 + '
        'T1(3帧中2帧)` 作为精度优先基线，同时保留 `Tiny + 640 + 阈值0.20` '
        '作为边缘端候选。最终型号必须在实际机器人算力上重新测速。',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    trial_rows, resolution_rows = build_resolution_tables()
    model_trial_rows, model_rows = build_model_tables()
    threshold_rows = read_csv(
        EXPERIMENT_ROOT / 'E3_thresholds' / 'threshold_metrics.csv')
    threshold_rows_s = read_csv(
        EXPERIMENT_ROOT / 'E3_thresholds_s' / 'threshold_metrics.csv')
    temporal_rows = read_csv(
        EXPERIMENT_ROOT / 'E4_temporal' / 'temporal_sweep.csv')

    write_csv(REPORT_ROOT / 'resolution_trials.csv', trial_rows)
    write_csv(REPORT_ROOT / 'resolution_summary.csv', resolution_rows)
    write_csv(REPORT_ROOT / 'model_trials.csv', model_trial_rows)
    write_csv(REPORT_ROOT / 'model_summary.csv', model_rows)
    plot_resolution_tradeoff(resolution_rows)
    plot_latency_breakdown(resolution_rows)
    plot_model_comparison(model_rows)
    plot_threshold_tradeoff(threshold_rows)
    plot_threshold_f1_models(threshold_rows, threshold_rows_s)
    plot_temporal_proxy(temporal_rows)
    (REPORT_ROOT / 'experiment_report.md').write_text(
        build_markdown(
            model_rows, resolution_rows, threshold_rows, threshold_rows_s,
            temporal_rows),
        encoding='utf-8')
    print(f'Wrote report to {REPORT_ROOT}')


if __name__ == '__main__':
    main()
