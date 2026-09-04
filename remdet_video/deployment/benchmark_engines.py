"""Benchmark RemDet TensorRT FP32/FP16 engines with Jetson telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATHS = {
    'fp32': DEPLOY_ROOT / 'engines/remdet_s_640_fp32.engine',
    'fp16': DEPLOY_ROOT / 'engines/remdet_s_640_fp16.engine',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--warmup-ms', type=int, default=2000)
    parser.add_argument('--cooldown', type=int, default=5)
    parser.add_argument('--telemetry-interval-ms', type=int, default=200)
    parser.add_argument(
        '--output', type=Path,
        default=DEPLOY_ROOT / 'results/benchmark_15w_pure_engine.json')
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        'count': len(values),
        'mean': float(statistics.fmean(values)),
        'min': float(ordered[0]),
        'max': float(ordered[-1]),
        'p95': float(ordered[p95_index]),
    }


def parse_distribution(output: str, label: str) -> dict | None:
    number = r'([0-9]+(?:\.[0-9]+)?)'
    pattern = (
        re.escape(label) + r':\s*min\s*=\s*' + number +
        r'\s*ms,\s*max\s*=\s*' + number +
        r'\s*ms,\s*mean\s*=\s*' + number +
        r'\s*ms,\s*median\s*=\s*' + number +
        r'\s*ms,\s*percentile\(95%\)\s*=\s*' + number + r'\s*ms')
    match = re.search(pattern, output)
    if not match:
        return None
    minimum, maximum, mean, median, p95 = map(float, match.groups())
    return {
        'min_ms': minimum,
        'max_ms': maximum,
        'mean_ms': mean,
        'median_ms': median,
        'p95_ms': p95,
    }


def parse_trtexec(output: str) -> dict:
    throughput_match = re.search(
        r'Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*qps', output)
    return {
        'throughput_qps': (
            float(throughput_match.group(1)) if throughput_match else None),
        'latency': parse_distribution(output, 'Latency'),
        'gpu_compute': parse_distribution(output, 'GPU Compute Time'),
        'enqueue_time': parse_distribution(output, 'Enqueue Time'),
        'passed': '&&&& PASSED TensorRT.trtexec' in output,
    }


def parse_tegrastats(path: Path) -> dict:
    ram_used = []
    gpu_load = []
    gpu_temperature = []
    junction_temperature = []
    input_power = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if match:
            ram_used.append(float(match.group(1)))
        match = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if match:
            gpu_load.append(float(match.group(1)))
        match = re.search(r'gpu@([0-9.]+)C', line)
        if match:
            gpu_temperature.append(float(match.group(1)))
        match = re.search(r'tj@([0-9.]+)C', line)
        if match:
            junction_temperature.append(float(match.group(1)))
        match = re.search(r'VDD_IN\s+(\d+)mW', line)
        if match:
            input_power.append(float(match.group(1)))
    return {
        'sample_count': max(
            len(ram_used), len(gpu_load), len(gpu_temperature),
            len(input_power)),
        'ram_used_mb': distribution(ram_used),
        'gpu_load_percent': distribution(gpu_load),
        'gpu_temperature_c': distribution(gpu_temperature),
        'junction_temperature_c': distribution(junction_temperature),
        'input_power_mw': distribution(input_power),
    }


def read_power_mode() -> str | None:
    try:
        result = subprocess.run(
            ['nvpmodel', '-q'], capture_output=True, text=True, timeout=5,
            check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout + result.stderr).strip()
    return text or None


def run_trial(
    precision: str,
    trial: int,
    engine: Path,
    results_dir: Path,
    duration: int,
    warmup_ms: int,
    telemetry_interval_ms: int,
) -> dict:
    trtexec_log = results_dir / f'benchmark_{precision}_trial{trial}.log'
    telemetry_log = results_dir / f'tegrastats_{precision}_trial{trial}.log'
    command = [
        'trtexec',
        f'--loadEngine={engine}',
        f'--warmUp={warmup_ms}',
        f'--duration={duration}',
        '--iterations=200',
        '--useCudaGraph',
        '--noDataTransfers',
        '--useSpinWait',
        '--percentile=95',
    ]
    telemetry = subprocess.Popen(
        [
            'tegrastats',
            '--interval', str(telemetry_interval_ms),
            '--logfile', str(telemetry_log),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    time.sleep(0.5)
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False)
    finally:
        elapsed = time.perf_counter() - started
        telemetry.terminate()
        try:
            telemetry.wait(timeout=5)
        except subprocess.TimeoutExpired:
            telemetry.kill()
            telemetry.wait(timeout=5)

    combined_output = result.stdout + result.stderr
    trtexec_log.write_text(combined_output, encoding='utf-8')
    parsed = parse_trtexec(combined_output)
    return {
        'precision': precision,
        'trial': trial,
        'command': command,
        'return_code': result.returncode,
        'wall_time_seconds': elapsed,
        'trtexec': parsed,
        'telemetry': parse_tegrastats(telemetry_log),
        'files': {
            'trtexec_log': str(trtexec_log),
            'tegrastats_log': str(telemetry_log),
        },
    }


def summarize_trials(trials: list[dict], precision: str) -> dict:
    selected = [trial for trial in trials if trial['precision'] == precision]
    throughput = [
        trial['trtexec']['throughput_qps'] for trial in selected
        if trial['trtexec']['throughput_qps'] is not None]
    gpu_mean = [
        trial['trtexec']['gpu_compute']['mean_ms'] for trial in selected
        if trial['trtexec']['gpu_compute'] is not None]
    gpu_p95 = [
        trial['trtexec']['gpu_compute']['p95_ms'] for trial in selected
        if trial['trtexec']['gpu_compute'] is not None]
    power = [
        trial['telemetry']['input_power_mw']['mean'] for trial in selected
        if trial['telemetry']['input_power_mw'] is not None]
    peak_temperature = [
        trial['telemetry']['gpu_temperature_c']['max'] for trial in selected
        if trial['telemetry']['gpu_temperature_c'] is not None]
    peak_ram = [
        trial['telemetry']['ram_used_mb']['max'] for trial in selected
        if trial['telemetry']['ram_used_mb'] is not None]
    return {
        'trial_count': len(selected),
        'all_trials_passed': all(
            trial['return_code'] == 0 and trial['trtexec']['passed']
            for trial in selected),
        'throughput_qps': distribution(throughput),
        'gpu_compute_mean_ms': distribution(gpu_mean),
        'gpu_compute_p95_ms': distribution(gpu_p95),
        'mean_input_power_mw': distribution(power),
        'peak_gpu_temperature_c': distribution(peak_temperature),
        'peak_ram_used_mb': distribution(peak_ram),
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


def main() -> None:
    args = parse_args()
    if args.trials <= 0 or args.duration <= 0 or args.warmup_ms < 0:
        raise ValueError('Trials/duration must be positive and warmup non-negative.')
    for engine in ENGINE_PATHS.values():
        if not engine.is_file():
            raise FileNotFoundError(engine)

    results_dir = args.output.resolve().parent
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'benchmark': 'pure TensorRT engine compute',
        'configuration': {
            'batch_size': 1,
            'input_shape': [1, 3, 640, 640],
            'power_mode_expected': '15W',
            'power_mode_query': read_power_mode(),
            'warmup_ms': args.warmup_ms,
            'duration_seconds_per_trial': args.duration,
            'trial_count_per_precision': args.trials,
            'use_cuda_graph': True,
            'data_transfers_included': False,
            'spin_wait': True,
            'telemetry_interval_ms': args.telemetry_interval_ms,
        },
        'environment': {
            'platform': platform.platform(),
            'python': platform.python_version(),
        },
        'engines': {
            precision: {
                'path': str(path),
                'size_bytes': path.stat().st_size,
                'sha256': sha256(path),
            } for precision, path in ENGINE_PATHS.items()
        },
        'trials': [],
    }

    sequence = [
        (precision, trial)
        for trial in range(1, args.trials + 1)
        for precision in ('fp32', 'fp16')
    ]
    for index, (precision, trial_number) in enumerate(sequence, start=1):
        print(
            f'[{index}/{len(sequence)}] {precision.upper()} trial '
            f'{trial_number}: warmup {args.warmup_ms} ms + '
            f'measure {args.duration} s',
            flush=True,
        )
        trial = run_trial(
            precision=precision,
            trial=trial_number,
            engine=ENGINE_PATHS[precision],
            results_dir=results_dir,
            duration=args.duration,
            warmup_ms=args.warmup_ms,
            telemetry_interval_ms=args.telemetry_interval_ms,
        )
        report['trials'].append(trial)
        write_report(args.output, report)
        throughput = trial['trtexec']['throughput_qps']
        gpu_compute = trial['trtexec']['gpu_compute']
        print(
            f'  passed={trial["trtexec"]["passed"]}, '
            f'throughput={throughput} qps, gpu_compute={gpu_compute}',
            flush=True,
        )
        if trial['return_code'] != 0 or not trial['trtexec']['passed']:
            raise SystemExit(
                f'trtexec failed; inspect {trial["files"]["trtexec_log"]}')
        if index != len(sequence) and args.cooldown > 0:
            print(f'  cooldown {args.cooldown} s', flush=True)
            time.sleep(args.cooldown)

    report['summary'] = {
        precision: summarize_trials(report['trials'], precision)
        for precision in ('fp32', 'fp16')
    }
    fp32_ms = report['summary']['fp32']['gpu_compute_mean_ms']
    fp16_ms = report['summary']['fp16']['gpu_compute_mean_ms']
    if fp32_ms and fp16_ms:
        report['comparison'] = {
            'fp16_speedup_vs_fp32': fp32_ms['mean'] / fp16_ms['mean'],
            'fp16_latency_reduction_percent': (
                1.0 - fp16_ms['mean'] / fp32_ms['mean']) * 100.0,
        }
    write_report(args.output, report)
    print(json.dumps({
        'summary': report['summary'],
        'comparison': report.get('comparison'),
        'report': str(args.output.resolve()),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
