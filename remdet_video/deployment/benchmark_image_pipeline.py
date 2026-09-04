"""Benchmark the complete RemDet FP16 single-image pipeline on Jetson.

The timed path is: OpenCV image decode -> letterbox/normalize -> TensorRT
inference including host/device transfers -> NumPy NMS/post-processing.
Model loading, drawing and image writing are reported separately because a
long-running video service does not perform those operations for every frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

from postprocess import VISDRONE_CLASSES, postprocess_detections
from preprocess import preprocess_bgr
from trt_runtime import TensorRTRunner


DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--engine', type=Path,
        default=DEPLOY_ROOT / 'engines/remdet_s_640_fp16.engine')
    parser.add_argument(
        '--image', type=Path,
        default=DEPLOY_ROOT / 'data/reference_image.jpg')
    parser.add_argument(
        '--reference-detections', type=Path,
        default=DEPLOY_ROOT / 'data/reference_detections.json')
    parser.add_argument('--score-threshold', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.7)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--cooldown', type=float, default=3.0)
    parser.add_argument('--telemetry-interval-ms', type=int, default=200)
    parser.add_argument(
        '--output', type=Path,
        default=DEPLOY_ROOT / 'results/image_pipeline_fp16_15w.json')
    parser.add_argument(
        '--output-image', type=Path,
        default=DEPLOY_ROOT / 'results/reference_image_fp16_detected.jpg')
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict:
    if not values:
        raise ValueError('Cannot summarize an empty measurement list.')
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': int(array.size),
        'mean': float(np.mean(array)),
        'std': float(np.std(array)),
        'min': float(np.min(array)),
        'median': float(np.median(array)),
        'p90': float(np.percentile(array, 90)),
        'p95': float(np.percentile(array, 95)),
        'p99': float(np.percentile(array, 99)),
        'max': float(np.max(array)),
    }


def parse_tegrastats(path: Path) -> dict:
    measurements = {
        'ram_used_mb': [],
        'gpu_load_percent': [],
        'gpu_temperature_c': [],
        'junction_temperature_c': [],
        'input_power_mw': [],
    }
    if not path.is_file():
        return {'sample_count': 0}
    patterns = {
        'ram_used_mb': r'RAM\s+(\d+)/(?:\d+)MB',
        'gpu_load_percent': r'GR3D_FREQ\s+(\d+)%',
        'gpu_temperature_c': r'gpu@([0-9.]+)C',
        'junction_temperature_c': r'tj@([0-9.]+)C',
        'input_power_mw': r'VDD_IN\s+(\d+)mW',
    }
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        for name, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                measurements[name].append(float(match.group(1)))
    result = {
        'sample_count': max((len(v) for v in measurements.values()), default=0)
    }
    for name, values in measurements.items():
        result[name] = distribution(values) if values else None
    return result


def read_power_mode() -> str | None:
    try:
        result = subprocess.run(
            ['nvpmodel', '-q'], capture_output=True, text=True,
            timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout + result.stderr).strip()
    return output or None


def class_counts(detections: list[dict]) -> dict[str, int]:
    return dict(Counter(item['class_name'] for item in detections))


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    rendered = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection['bbox']]
        class_id = int(detection['class_id'])
        color = (
            int((37 * class_id + 80) % 256),
            int((17 * class_id + 170) % 256),
            int((97 * class_id + 40) % 256),
        )
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['class_name']} {detection['score']:.2f}"
        text_y = max(14, y1 - 4)
        cv2.putText(
            rendered, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, color, 1, cv2.LINE_AA)
    return rendered


def run_frame(
    runner: TensorRTRunner,
    image_path: Path,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[dict[str, float], list[dict], bool]:
    start = time.perf_counter_ns()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    after_decode = time.perf_counter_ns()
    if image is None:
        raise RuntimeError(f'OpenCV could not read {image_path}')

    tensor, meta = preprocess_bgr(image, 640)
    after_preprocess = time.perf_counter_ns()
    outputs = runner.run({'images': tensor})
    after_inference = time.perf_counter_ns()
    finite_outputs = bool(
        np.isfinite(outputs['boxes']).all() and
        np.isfinite(outputs['scores']).all())
    detections = postprocess_detections(
        outputs['boxes'], outputs['scores'], meta,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    after_postprocess = time.perf_counter_ns()

    to_ms = 1.0 / 1_000_000.0
    timings = {
        'decode_ms': (after_decode - start) * to_ms,
        'preprocess_ms': (after_preprocess - after_decode) * to_ms,
        'inference_with_transfers_ms': (
            after_inference - after_preprocess) * to_ms,
        'postprocess_nms_ms': (
            after_postprocess - after_inference) * to_ms,
        'end_to_end_ms': (after_postprocess - start) * to_ms,
    }
    return timings, detections, finite_outputs


def summarize_trial(measurements: dict[str, list[float]]) -> dict:
    summary = {name: distribution(values)
               for name, values in measurements.items()}
    summary['effective_fps_from_mean_e2e'] = (
        1000.0 / summary['end_to_end_ms']['mean'])
    return summary


def aggregate_trials(trials: list[dict]) -> dict:
    metrics = (
        'decode_ms', 'preprocess_ms', 'inference_with_transfers_ms',
        'postprocess_nms_ms', 'end_to_end_ms')
    result = {}
    for metric in metrics:
        means = [trial['timings'][metric]['mean'] for trial in trials]
        p95s = [trial['timings'][metric]['p95'] for trial in trials]
        result[metric] = {
            'mean_of_trial_means': float(statistics.fmean(means)),
            'mean_of_trial_p95s': float(statistics.fmean(p95s)),
            'trial_means': means,
            'trial_p95s': p95s,
        }
    mean_e2e = result['end_to_end_ms']['mean_of_trial_means']
    result['effective_fps_from_mean_e2e'] = 1000.0 / mean_e2e
    return result


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.trials <= 0:
        raise ValueError('Warmup must be non-negative; iterations/trials positive.')
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError('Score threshold must be between 0 and 1.')
    for required in (args.engine, args.image, args.reference_detections):
        if not required.is_file():
            raise FileNotFoundError(required)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    reference_payload = json.loads(
        args.reference_detections.read_text(encoding='utf-8'))
    reference_visible = [
        item for item in reference_payload['onnx']
        if item['score'] >= args.score_threshold
    ]

    print(f'Loading FP16 engine: {args.engine}', flush=True)
    load_started = time.perf_counter_ns()
    runner = TensorRTRunner(args.engine)
    engine_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000.0
    print(f'Engine loaded in {engine_load_ms:.2f} ms', flush=True)

    trials = []
    last_detections: list[dict] = []
    all_outputs_finite = True
    try:
        print(f'Warmup: {args.warmup} complete frames', flush=True)
        for _ in range(args.warmup):
            _, last_detections, finite = run_frame(
                runner, args.image, args.score_threshold, args.iou_threshold)
            all_outputs_finite = all_outputs_finite and finite

        for trial_number in range(1, args.trials + 1):
            print(
                f'[{trial_number}/{args.trials}] Measuring '
                f'{args.iterations} complete frames...', flush=True)
            telemetry_path = args.output.parent / (
                f'tegrastats_image_pipeline_trial{trial_number}.log')
            telemetry_path.unlink(missing_ok=True)
            telemetry = subprocess.Popen(
                ['tegrastats', '--interval', str(args.telemetry_interval_ms),
                 '--logfile', str(telemetry_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            time.sleep(0.5)
            measurements = {
                'decode_ms': [],
                'preprocess_ms': [],
                'inference_with_transfers_ms': [],
                'postprocess_nms_ms': [],
                'end_to_end_ms': [],
            }
            try:
                for _ in range(args.iterations):
                    timings, last_detections, finite = run_frame(
                        runner, args.image, args.score_threshold,
                        args.iou_threshold)
                    all_outputs_finite = all_outputs_finite and finite
                    for name, value in timings.items():
                        measurements[name].append(value)
            finally:
                telemetry.terminate()
                try:
                    telemetry.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    telemetry.kill()
                    telemetry.wait(timeout=5)

            trial_summary = summarize_trial(measurements)
            trial = {
                'trial': trial_number,
                'timings': trial_summary,
                'telemetry': parse_tegrastats(telemetry_path),
                'telemetry_log': str(telemetry_path.resolve()),
            }
            trials.append(trial)
            print(
                '  E2E mean={:.3f} ms, P95={:.3f} ms, FPS={:.2f}, '
                'detections={}'.format(
                    trial_summary['end_to_end_ms']['mean'],
                    trial_summary['end_to_end_ms']['p95'],
                    trial_summary['effective_fps_from_mean_e2e'],
                    len(last_detections)),
                flush=True)
            if trial_number != args.trials and args.cooldown > 0:
                print(f'  Cooldown {args.cooldown:g} s', flush=True)
                time.sleep(args.cooldown)
    finally:
        runner.close()

    source_image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    render_started = time.perf_counter_ns()
    rendered = draw_detections(source_image, last_detections)
    render_ms = (time.perf_counter_ns() - render_started) / 1_000_000.0
    write_started = time.perf_counter_ns()
    write_succeeded = bool(cv2.imwrite(str(args.output_image), rendered))
    write_ms = (time.perf_counter_ns() - write_started) / 1_000_000.0

    candidate_counts = class_counts(last_detections)
    reference_counts = class_counts(reference_visible)
    count_delta = abs(len(last_detections) - len(reference_visible))
    class_names = set(candidate_counts) | set(reference_counts)
    maximum_class_delta = max(
        (abs(candidate_counts.get(name, 0) - reference_counts.get(name, 0))
         for name in class_names),
        default=0,
    )
    checks = {
        'finite_outputs': all_outputs_finite,
        'visible_detection_count_within_fp16_tolerance': count_delta <= 1,
        'visible_class_counts_within_fp16_tolerance': (
            maximum_class_delta <= 1),
        'output_image_written': write_succeeded and args.output_image.is_file(),
    }
    aggregate = aggregate_trials(trials)
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'passed': all(checks.values()),
        'checks': checks,
        'benchmark': (
            'single-image complete pipeline excluding drawing and file output'),
        'configuration': {
            'batch_size': 1,
            'input_shape': [1, 3, 640, 640],
            'score_threshold': args.score_threshold,
            'iou_threshold': args.iou_threshold,
            'warmup_frames': args.warmup,
            'iterations_per_trial': args.iterations,
            'trials': args.trials,
            'power_mode_query': read_power_mode(),
            'telemetry_interval_ms': args.telemetry_interval_ms,
        },
        'environment': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'tensorrt': trt.__version__,
            'opencv': cv2.__version__,
            'numpy': np.__version__,
        },
        'files': {
            'engine': str(args.engine.resolve()),
            'engine_sha256': sha256(args.engine),
            'image': str(args.image.resolve()),
            'image_sha256': sha256(args.image),
            'rendered_image': str(args.output_image.resolve()),
        },
        'startup_engine_load_ms': engine_load_ms,
        'trials': trials,
        'summary': aggregate,
        'rendering_not_in_e2e': {
            'draw_boxes_ms': render_ms,
            'write_jpeg_ms': write_ms,
        },
        'detections': {
            'reference_count': len(reference_visible),
            'candidate_count': len(last_detections),
            'reference_class_counts': reference_counts,
            'candidate_class_counts': candidate_counts,
            'count_delta': count_delta,
            'maximum_class_count_delta': maximum_class_delta,
            'items': last_detections,
        },
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'passed': report['passed'],
        'checks': checks,
        'summary': aggregate,
        'detection_count': len(last_detections),
        'class_counts': candidate_counts,
        'rendered_image': str(args.output_image.resolve()),
        'report': str(args.output.resolve()),
    }, indent=2, ensure_ascii=False), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
