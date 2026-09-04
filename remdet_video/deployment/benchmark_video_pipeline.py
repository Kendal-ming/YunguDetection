"""Benchmark the complete RemDet FP16 video pipeline on Jetson.

The script always benchmarks OpenCV's normal video reader and automatically
adds a Jetson GStreamer/NVDEC reader when the local OpenCV build supports it.
It loops a short clip to obtain enough timing samples, while excluding clip
reopen, drawing and output encoding from the measured detection pipeline.
"""

from __future__ import annotations

import argparse
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

from benchmark_image_pipeline import (
    distribution,
    draw_detections,
    parse_tegrastats,
    read_power_mode,
    sha256,
)
from postprocess import postprocess_detections
from preprocess import preprocess_bgr
from trt_runtime import TensorRTRunner


DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--engine', type=Path,
        default=DEPLOY_ROOT / 'engines/remdet_s_640_fp16.engine')
    parser.add_argument(
        '--video', type=Path,
        default=DEPLOY_ROOT / 'data/demo_720p_h264.mp4')
    parser.add_argument('--score-threshold', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.7)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--cooldown', type=float, default=3.0)
    parser.add_argument('--telemetry-interval-ms', type=int, default=200)
    parser.add_argument(
        '--enable-nvdec', action='store_true',
        help=(
            'Opt in to the experimental GStreamer/NVDEC path. It is disabled '
            'by default because incompatible Jetson multimedia plugins can '
            'terminate the Python process during capability probing.'))
    parser.add_argument(
        '--output', type=Path,
        default=DEPLOY_ROOT / 'results/video_pipeline_fp16_15w.json')
    parser.add_argument(
        '--output-video', type=Path,
        default=DEPLOY_ROOT / 'results/demo_720p_fp16_detected.mp4')
    return parser.parse_args()


def opencv_has_gstreamer() -> bool:
    match = re.search(
        r'^\s*GStreamer:\s+(\S+)', cv2.getBuildInformation(), re.MULTILINE)
    return bool(match and match.group(1).upper() == 'YES')


def gstreamer_pipeline(path: Path) -> str:
    escaped = str(path.resolve()).replace('\\', '\\\\').replace('"', '\\"')
    return (
        f'filesrc location="{escaped}" ! qtdemux name=demux '
        'demux.video_0 ! queue ! h264parse ! nvv4l2decoder ! '
        'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! '
        'video/x-raw,format=BGR ! appsink max-buffers=2 drop=false sync=false'
    )


def open_capture(mode: str, path: Path) -> cv2.VideoCapture:
    if mode == 'opencv':
        capture = cv2.VideoCapture(str(path))
    elif mode == 'gstreamer_nvdec':
        capture = cv2.VideoCapture(gstreamer_pipeline(path), cv2.CAP_GSTREAMER)
    else:
        raise ValueError(f'Unknown video mode: {mode}')
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f'Could not open {path} using {mode}')
    return capture


def probe_mode(mode: str, path: Path) -> tuple[bool, str | None]:
    try:
        capture = open_capture(mode, path)
        try:
            for _ in range(3):
                ok, frame = capture.read()
                if not ok or frame is None:
                    return False, 'capture opened but could not decode three frames'
        finally:
            capture.release()
    except Exception as error:
        return False, str(error)
    return True, None


def read_successful_frame(
    capture: cv2.VideoCapture,
    mode: str,
    path: Path,
) -> tuple[cv2.VideoCapture, np.ndarray, int, int]:
    """Read one frame, reopening a looped clip outside the returned timing."""
    for _ in range(3):
        started = time.perf_counter_ns()
        ok, frame = capture.read()
        ended = time.perf_counter_ns()
        if ok and frame is not None:
            return capture, frame, started, ended
        capture.release()
        capture = open_capture(mode, path)
    capture.release()
    raise RuntimeError(f'Could not decode a frame from {path} using {mode}')


def process_frame(
    runner: TensorRTRunner,
    frame: np.ndarray,
    decode_started: int,
    decode_ended: int,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[dict[str, float], list[dict], bool]:
    tensor, meta = preprocess_bgr(frame, 640)
    after_preprocess = time.perf_counter_ns()
    outputs = runner.run({'images': tensor})
    after_inference = time.perf_counter_ns()
    finite = bool(
        np.isfinite(outputs['boxes']).all() and
        np.isfinite(outputs['scores']).all())
    detections = postprocess_detections(
        outputs['boxes'], outputs['scores'], meta,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold)
    after_postprocess = time.perf_counter_ns()
    to_ms = 1.0 / 1_000_000.0
    return {
        'decode_ms': (decode_ended - decode_started) * to_ms,
        'preprocess_ms': (after_preprocess - decode_ended) * to_ms,
        'inference_with_transfers_ms': (
            after_inference - after_preprocess) * to_ms,
        'postprocess_nms_ms': (
            after_postprocess - after_inference) * to_ms,
        'end_to_end_ms': (after_postprocess - decode_started) * to_ms,
    }, detections, finite


def run_trial(
    runner: TensorRTRunner,
    mode: str,
    trial_number: int,
    args: argparse.Namespace,
) -> dict:
    capture = open_capture(mode, args.video)
    telemetry_path = args.output.parent / (
        f'tegrastats_video_{mode}_trial{trial_number}.log')
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
    total_detections = 0
    class_counter: Counter[str] = Counter()
    all_finite = True
    try:
        for _ in range(args.frames):
            capture, frame, decode_started, decode_ended = read_successful_frame(
                capture, mode, args.video)
            timings, detections, finite = process_frame(
                runner, frame, decode_started, decode_ended,
                args.score_threshold, args.iou_threshold)
            all_finite = all_finite and finite
            total_detections += len(detections)
            class_counter.update(item['class_name'] for item in detections)
            for name, value in timings.items():
                measurements[name].append(value)
    finally:
        capture.release()
        telemetry.terminate()
        try:
            telemetry.wait(timeout=5)
        except subprocess.TimeoutExpired:
            telemetry.kill()
            telemetry.wait(timeout=5)

    timing_summary = {
        name: distribution(values) for name, values in measurements.items()
    }
    timing_summary['effective_fps_from_mean_e2e'] = (
        1000.0 / timing_summary['end_to_end_ms']['mean'])
    return {
        'mode': mode,
        'trial': trial_number,
        'passed': all_finite and len(measurements['end_to_end_ms']) == args.frames,
        'frames': args.frames,
        'timings': timing_summary,
        'detections_per_frame_mean': total_detections / args.frames,
        'class_counts_all_frames': dict(class_counter),
        'telemetry': parse_tegrastats(telemetry_path),
        'telemetry_log': str(telemetry_path.resolve()),
    }


def aggregate_mode(trials: list[dict], mode: str) -> dict:
    selected = [trial for trial in trials if trial['mode'] == mode]
    metrics = (
        'decode_ms', 'preprocess_ms', 'inference_with_transfers_ms',
        'postprocess_nms_ms', 'end_to_end_ms')
    result = {
        'trial_count': len(selected),
        'all_trials_passed': all(trial['passed'] for trial in selected),
    }
    for metric in metrics:
        means = [trial['timings'][metric]['mean'] for trial in selected]
        p95s = [trial['timings'][metric]['p95'] for trial in selected]
        result[metric] = {
            'mean_of_trial_means': float(statistics.fmean(means)),
            'mean_of_trial_p95s': float(statistics.fmean(p95s)),
            'trial_means': means,
        }
    e2e = result['end_to_end_ms']['mean_of_trial_means']
    result['effective_fps_from_mean_e2e'] = 1000.0 / e2e

    for field, source, statistic in (
        ('mean_input_power_mw', 'input_power_mw', 'mean'),
        ('peak_gpu_temperature_c', 'gpu_temperature_c', 'max'),
        ('peak_ram_used_mb', 'ram_used_mb', 'max'),
    ):
        values = [
            trial['telemetry'][source][statistic]
            for trial in selected
            if trial['telemetry'].get(source) is not None
        ]
        result[field] = distribution(values) if values else None
    return result


def write_annotated_video(
    runner: TensorRTRunner,
    args: argparse.Namespace,
    source_metadata: dict,
) -> dict:
    capture = open_capture('opencv', args.video)
    width = int(source_metadata['width'])
    height = int(source_metadata['height'])
    fps = float(source_metadata['fps']) or 30.0
    frame_count = int(source_metadata['frame_count'])
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f'Could not create {args.output_video}')
    written = 0
    started = time.perf_counter_ns()
    try:
        while written < frame_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            tensor, meta = preprocess_bgr(frame, 640)
            outputs = runner.run({'images': tensor})
            detections = postprocess_detections(
                outputs['boxes'], outputs['scores'], meta,
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold)
            writer.write(draw_detections(frame, detections))
            written += 1
    finally:
        capture.release()
        writer.release()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        'path': str(args.output_video.resolve()),
        'frames_written': written,
        'elapsed_ms_including_draw_and_encode': elapsed_ms,
        'file_written': args.output_video.is_file() and written > 0,
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.frames <= 0 or args.trials <= 0:
        raise ValueError('Warmup must be non-negative; frames/trials positive.')
    for path in (args.engine, args.video):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metadata_capture = open_capture('opencv', args.video)
    source_metadata = {
        'width': int(metadata_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(metadata_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': float(metadata_capture.get(cv2.CAP_PROP_FPS)),
        'frame_count': int(metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fourcc': ''.join(chr((int(metadata_capture.get(cv2.CAP_PROP_FOURCC))
                               >> (8 * index)) & 255) for index in range(4)),
    }
    metadata_capture.release()

    mode_probe = {'opencv': {'available': True, 'reason': None}}
    modes = ['opencv']
    if args.enable_nvdec and opencv_has_gstreamer():
        available, reason = probe_mode('gstreamer_nvdec', args.video)
        mode_probe['gstreamer_nvdec'] = {
            'available': available, 'reason': reason}
        if available:
            modes.append('gstreamer_nvdec')
    elif not args.enable_nvdec:
        mode_probe['gstreamer_nvdec'] = {
            'available': False,
            'reason': (
                'Disabled by default; pass --enable-nvdec only after the '
                'Jetson multimedia stack has been validated separately'),
        }
    else:
        mode_probe['gstreamer_nvdec'] = {
            'available': False,
            'reason': 'OpenCV was built without GStreamer support',
        }

    print(f'Video: {source_metadata}', flush=True)
    print(f'Decoder modes: {mode_probe}', flush=True)
    load_started = time.perf_counter_ns()
    runner = TensorRTRunner(args.engine)
    engine_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000.0
    trials = []
    try:
        for mode in modes:
            print(f'Warmup {mode}: {args.warmup} frames', flush=True)
            capture = open_capture(mode, args.video)
            try:
                for _ in range(args.warmup):
                    capture, frame, begin, decoded = read_successful_frame(
                        capture, mode, args.video)
                    process_frame(
                        runner, frame, begin, decoded,
                        args.score_threshold, args.iou_threshold)
            finally:
                capture.release()

            for trial_number in range(1, args.trials + 1):
                print(
                    f'[{mode} {trial_number}/{args.trials}] '
                    f'{args.frames} frames...', flush=True)
                trial = run_trial(runner, mode, trial_number, args)
                trials.append(trial)
                e2e = trial['timings']['end_to_end_ms']
                print(
                    f'  passed={trial["passed"]}, mean={e2e["mean"]:.3f} '
                    f'ms, P95={e2e["p95"]:.3f} ms, '
                    f'FPS={trial["timings"]["effective_fps_from_mean_e2e"]:.2f}',
                    flush=True)
                if trial_number != args.trials and args.cooldown > 0:
                    time.sleep(args.cooldown)

        annotated = write_annotated_video(runner, args, source_metadata)
    finally:
        runner.close()

    summary = {mode: aggregate_mode(trials, mode) for mode in modes}
    comparison = None
    if 'gstreamer_nvdec' in summary:
        cpu_ms = summary['opencv']['end_to_end_ms']['mean_of_trial_means']
        nvdec_ms = summary['gstreamer_nvdec'][
            'end_to_end_ms']['mean_of_trial_means']
        comparison = {
            'nvdec_speedup_vs_opencv': cpu_ms / nvdec_ms,
            'nvdec_latency_reduction_percent': (1.0 - nvdec_ms / cpu_ms) * 100,
        }

    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'passed': all(item['all_trials_passed'] for item in summary.values())
                  and annotated['file_written'],
        'benchmark': (
            'video decode + preprocess + TensorRT with transfers + NumPy NMS; '
            'drawing and output encoding excluded from timing'),
        'configuration': {
            'warmup_frames_per_mode': args.warmup,
            'measured_frames_per_trial': args.frames,
            'trials_per_mode': args.trials,
            'score_threshold': args.score_threshold,
            'iou_threshold': args.iou_threshold,
            'power_mode_query': read_power_mode(),
        },
        'environment': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'tensorrt': trt.__version__,
            'opencv': cv2.__version__,
            'opencv_gstreamer': opencv_has_gstreamer(),
        },
        'files': {
            'engine': str(args.engine.resolve()),
            'engine_sha256': sha256(args.engine),
            'video': str(args.video.resolve()),
            'video_sha256': sha256(args.video),
        },
        'source_video': source_metadata,
        'mode_probe': mode_probe,
        'startup_engine_load_ms': engine_load_ms,
        'trials': trials,
        'summary': summary,
        'comparison': comparison,
        'annotated_video': annotated,
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({
        'passed': report['passed'],
        'source_video': source_metadata,
        'mode_probe': mode_probe,
        'summary': summary,
        'comparison': comparison,
        'annotated_video': annotated,
        'report': str(args.output.resolve()),
    }, indent=2, ensure_ascii=False), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
