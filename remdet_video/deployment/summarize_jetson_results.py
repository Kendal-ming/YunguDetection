"""Create final RemDet Windows-versus-Jetson reports and plots."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / 'work_dirs/deployment/jetson_results'
WINDOWS = {
    'device': 'NVIDIA GeForce RTX 5080',
    'framework': 'PyTorch FP32',
    'model_latency_ms': 7.566,
    'end_to_end_latency_ms': 10.768,
    'end_to_end_fps': 92.864,
    'bbox_mAP': 0.247,
    'bbox_mAP_50': 0.415,
    'bbox_mAP_75': 0.250,
    'bbox_mAP_s': 0.154,
    'bbox_mAP_m': 0.367,
    'bbox_mAP_l': 0.470,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def fmt(value: float | None, digits: int = 3) -> str:
    return 'N/A' if value is None else f'{value:.{digits}f}'


def choose_video_mode(video: dict) -> tuple[str, dict]:
    summary = video['summary']
    return min(
        summary.items(),
        key=lambda item: item[1]['end_to_end_ms']['mean_of_trial_means'])


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        'platform', 'workload', 'precision', 'latency_mean_ms',
        'latency_p95_ms', 'fps', 'power_w', 'peak_gpu_temp_c',
        'peak_ram_mb']
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_figure(
    path: Path,
    pure: dict,
    image: dict,
    video_mode: str,
    video_best: dict,
    evaluation: dict,
) -> None:
    labels = [
        'RTX 5080\nPyTorch model',
        'Jetson\nTRT FP16 model',
        'Jetson\nimage E2E',
        f'Jetson\nvideo E2E\n({video_mode})',
    ]
    latency = [
        WINDOWS['model_latency_ms'],
        pure['summary']['fp16']['gpu_compute_mean_ms']['mean'],
        image['summary']['end_to_end_ms']['mean_of_trial_means'],
        video_best['end_to_end_ms']['mean_of_trial_means'],
    ]
    fps = [1000.0 / value for value in latency]

    image_breakdown = [
        image['summary'][name]['mean_of_trial_means']
        for name in (
            'decode_ms', 'preprocess_ms',
            'inference_with_transfers_ms', 'postprocess_nms_ms')]
    video_breakdown = [
        video_best[name]['mean_of_trial_means']
        for name in (
            'decode_ms', 'preprocess_ms',
            'inference_with_transfers_ms', 'postprocess_nms_ms')]
    stage_labels = ['Decode', 'Preprocess', 'Inference', 'NMS']

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = ['#507DBC', '#2A9D8F', '#F4A261', '#E76F51']

    axes[0, 0].bar(labels, latency, color=colors)
    axes[0, 0].set_ylabel('Milliseconds (lower is better)')
    axes[0, 0].set_title('Latency comparison')
    axes[0, 0].grid(axis='y', alpha=0.25)
    for index, value in enumerate(latency):
        axes[0, 0].text(index, value, f'{value:.2f}', ha='center', va='bottom')

    axes[0, 1].bar(labels, fps, color=colors)
    axes[0, 1].axhline(30, color='#444444', linestyle='--', label='30 FPS')
    axes[0, 1].set_ylabel('Frames per second (higher is better)')
    axes[0, 1].set_title('Throughput comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.25)
    for index, value in enumerate(fps):
        axes[0, 1].text(index, value, f'{value:.1f}', ha='center', va='bottom')

    x_positions = [0, 1]
    bottoms = [0.0, 0.0]
    for stage, image_value, video_value, color in zip(
            stage_labels, image_breakdown, video_breakdown, colors):
        values = [image_value, video_value]
        axes[1, 0].bar(
            x_positions, values, bottom=bottoms, label=stage, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axes[1, 0].set_xticks(x_positions, ['Repeated JPEG', f'Video ({video_mode})'])
    axes[1, 0].set_ylabel('Milliseconds')
    axes[1, 0].set_title('Jetson end-to-end latency breakdown')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.25)

    metric_names = ['mAP', 'AP50', 'AP75', 'AP-small']
    baseline = [
        WINDOWS['bbox_mAP'], WINDOWS['bbox_mAP_50'],
        WINDOWS['bbox_mAP_75'], WINDOWS['bbox_mAP_s']]
    jetson = [
        evaluation['metrics']['bbox_mAP'],
        evaluation['metrics']['bbox_mAP_50'],
        evaluation['metrics']['bbox_mAP_75'],
        evaluation['metrics']['bbox_mAP_s'],
    ]
    xs = list(range(len(metric_names)))
    width = 0.36
    axes[1, 1].bar(
        [value - width / 2 for value in xs], baseline, width,
        label='Windows PyTorch FP32', color='#507DBC')
    axes[1, 1].bar(
        [value + width / 2 for value in xs], jetson, width,
        label='Jetson TensorRT FP16', color='#2A9D8F')
    axes[1, 1].set_xticks(xs, metric_names)
    axes[1, 1].set_ylim(0, max(baseline + jetson) * 1.25)
    axes[1, 1].set_title('VisDrone accuracy retention')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.25)

    fig.suptitle('RemDet-S 640 deployment summary', fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = args.results_dir.resolve()
    pure = load_json(results / 'benchmark_15w_pure_engine.json')
    image = load_json(results / 'image_pipeline_fp16_15w.json')
    video = load_json(results / 'video_pipeline_fp16_15w.json')
    dataset = load_json(results / 'visdrone_val_inference_trt_fp16_15w.json')
    evaluation = load_json(results / 'visdrone_val_coco_eval.json')
    video_mode, video_best = choose_video_mode(video)

    pure_fp16 = pure['summary']['fp16']
    rows = [
        {
            'platform': 'RTX 5080',
            'workload': 'PyTorch model',
            'precision': 'FP32',
            'latency_mean_ms': WINDOWS['model_latency_ms'],
            'latency_p95_ms': '',
            'fps': 1000.0 / WINDOWS['model_latency_ms'],
            'power_w': '', 'peak_gpu_temp_c': '', 'peak_ram_mb': '',
        },
        {
            'platform': 'Jetson Orin NX 15W',
            'workload': 'TensorRT pure model',
            'precision': 'FP16',
            'latency_mean_ms': pure_fp16['gpu_compute_mean_ms']['mean'],
            'latency_p95_ms': pure_fp16['gpu_compute_p95_ms']['mean'],
            'fps': pure_fp16['throughput_qps']['mean'],
            'power_w': pure_fp16['mean_input_power_mw']['mean'] / 1000.0,
            'peak_gpu_temp_c': pure_fp16['peak_gpu_temperature_c']['max'],
            'peak_ram_mb': pure_fp16['peak_ram_used_mb']['max'],
        },
        {
            'platform': 'Jetson Orin NX 15W',
            'workload': 'Repeated JPEG E2E',
            'precision': 'FP16',
            'latency_mean_ms': image['summary']['end_to_end_ms'][
                'mean_of_trial_means'],
            'latency_p95_ms': image['summary']['end_to_end_ms'][
                'mean_of_trial_p95s'],
            'fps': image['summary']['effective_fps_from_mean_e2e'],
            'power_w': '', 'peak_gpu_temp_c': '', 'peak_ram_mb': '',
        },
        {
            'platform': 'Jetson Orin NX 15W',
            'workload': f'Video E2E ({video_mode})',
            'precision': 'FP16',
            'latency_mean_ms': video_best['end_to_end_ms'][
                'mean_of_trial_means'],
            'latency_p95_ms': video_best['end_to_end_ms'][
                'mean_of_trial_p95s'],
            'fps': video_best['effective_fps_from_mean_e2e'],
            'power_w': (
                video_best['mean_input_power_mw']['mean'] / 1000.0
                if video_best.get('mean_input_power_mw') else ''),
            'peak_gpu_temp_c': (
                video_best['peak_gpu_temperature_c']['max']
                if video_best.get('peak_gpu_temperature_c') else ''),
            'peak_ram_mb': (
                video_best['peak_ram_used_mb']['max']
                if video_best.get('peak_ram_used_mb') else ''),
        },
        {
            'platform': 'Jetson Orin NX 15W',
            'workload': 'VisDrone val E2E',
            'precision': 'FP16',
            'latency_mean_ms': dataset['timings']['end_to_end_ms']['mean'],
            'latency_p95_ms': dataset['timings']['end_to_end_ms']['p95'],
            'fps': dataset['timings']['effective_fps_from_mean_e2e'],
            'power_w': dataset['telemetry']['input_power_mw']['mean'] / 1000.0,
            'peak_gpu_temp_c': dataset['telemetry'][
                'gpu_temperature_c']['max'],
            'peak_ram_mb': dataset['telemetry']['ram_used_mb']['max'],
        },
    ]

    summary = {
        'created_at': datetime.now().astimezone().isoformat(),
        'passed': bool(
            pure['summary']['fp16']['all_trials_passed'] and
            image['passed'] and video['passed'] and dataset['passed'] and
            evaluation['passed']),
        'windows_baseline': WINDOWS,
        'jetson': {
            'pure_fp16': pure_fp16,
            'image_pipeline': image['summary'],
            'best_video_mode': video_mode,
            'video_pipeline': video_best,
            'visdrone_inference': dataset['timings'],
            'visdrone_accuracy': evaluation['metrics'],
            'accuracy_difference': evaluation[
                'difference_jetson_fp16_minus_windows_pytorch'],
        },
    }
    summary_path = results / 'jetson_deployment_summary.json'
    csv_path = results / 'jetson_performance_comparison.csv'
    plot_path = results / 'jetson_deployment_comparison.png'
    report_path = results / 'jetson_deployment_report.md'
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    write_csv(csv_path, rows)
    create_figure(plot_path, pure, image, video_mode, video_best, evaluation)

    metrics = evaluation['metrics']
    report = f"""# RemDet-S 640 Jetson Orin NX 部署报告

生成时间：{summary['created_at']}

## 结论

- 总体验证状态：{'通过' if summary['passed'] else '需要检查'}。
- Jetson 15W TensorRT FP16 纯模型：{fmt(pure_fp16['gpu_compute_mean_ms']['mean'])} ms，{fmt(pure_fp16['throughput_qps']['mean'], 2)} FPS。
- Jetson 最佳视频模式：`{video_mode}`，端到端 {fmt(video_best['end_to_end_ms']['mean_of_trial_means'])} ms，{fmt(video_best['effective_fps_from_mean_e2e'], 2)} FPS。
- Jetson VisDrone mAP：{fmt(metrics['bbox_mAP'])}；Windows PyTorch 基准：{fmt(WINDOWS['bbox_mAP'])}。
- mAP 差值：{fmt(metrics['bbox_mAP'] - WINDOWS['bbox_mAP'])}。

## 精度对比

| 指标 | Windows PyTorch FP32 | Jetson TensorRT FP16 | 差值 |
|---|---:|---:|---:|
| mAP | {fmt(WINDOWS['bbox_mAP'])} | {fmt(metrics['bbox_mAP'])} | {fmt(metrics['bbox_mAP'] - WINDOWS['bbox_mAP'])} |
| AP50 | {fmt(WINDOWS['bbox_mAP_50'])} | {fmt(metrics['bbox_mAP_50'])} | {fmt(metrics['bbox_mAP_50'] - WINDOWS['bbox_mAP_50'])} |
| AP75 | {fmt(WINDOWS['bbox_mAP_75'])} | {fmt(metrics['bbox_mAP_75'])} | {fmt(metrics['bbox_mAP_75'] - WINDOWS['bbox_mAP_75'])} |
| AP-small | {fmt(WINDOWS['bbox_mAP_s'])} | {fmt(metrics['bbox_mAP_s'])} | {fmt(metrics['bbox_mAP_s'] - WINDOWS['bbox_mAP_s'])} |
| AP-medium | {fmt(WINDOWS['bbox_mAP_m'])} | {fmt(metrics['bbox_mAP_m'])} | {fmt(metrics['bbox_mAP_m'] - WINDOWS['bbox_mAP_m'])} |
| AP-large | {fmt(WINDOWS['bbox_mAP_l'])} | {fmt(metrics['bbox_mAP_l'])} | {fmt(metrics['bbox_mAP_l'] - WINDOWS['bbox_mAP_l'])} |

## 文件

- 图表：`jetson_deployment_comparison.png`
- 性能表：`jetson_performance_comparison.csv`
- 机器可读汇总：`jetson_deployment_summary.json`
- 完整COCO评估：`visdrone_val_coco_eval.json`
"""
    report_path.write_text(report, encoding='utf-8')
    print(json.dumps({
        'passed': summary['passed'],
        'best_video_mode': video_mode,
        'files': {
            'summary': str(summary_path),
            'csv': str(csv_path),
            'plot': str(plot_path),
            'report': str(report_path),
        },
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
