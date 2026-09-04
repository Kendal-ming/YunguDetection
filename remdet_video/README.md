# RemDet video experiments

This directory adds video inference, latency measurement and temporal
experiments without replacing the original RemDet model implementation.

## Design rules

- Original files under `mmdet/` and `config_remdet/remdet/` remain the baseline.
- Video inference loads the model once and processes frames with batch size 1.
- Safe checkpoints have no dataset metadata, so `RemDetDetector` explicitly
  installs the ten VisDrone class names.
- CUDA is synchronized around `model.test_step` for honest wall-clock latency.
- Low-threshold predictions are cached once and reused for threshold and
  temporal sweeps.
- Unlabelled-video stability statistics are proxy metrics, not accuracy.

## Current experiment entry points

Run these commands after `conda activate remdet5080` from the RemDet root.

Set the VisDrone data directory once in the current PowerShell window. The
repository otherwise looks for `../datasets/VisDrone2019-DET-COCO` relative to
the RemDet root.

```powershell
$env:VISDRONE_DATA_ROOT = 'C:\Users\xh\Desktop\WindyLab\datasets\VisDrone2019-DET-COCO'
```

```powershell
python remdet_video\tools\run_video_experiment.py `
  --experiment tiny_640 `
  --video demo\demo.mp4 `
  --config config_remdet\remdet\remdet_tiny-300e_coco.py `
  --checkpoint checkpoints\remdet_tiny_weights_only.pth `
  --output-dir work_dirs\video_experiments\tiny_640 `
  --warmup 50 --repeats 10

python remdet_video\tools\evaluate_thresholds.py `
  --predictions work_dirs\video_experiments\E0_accuracy\visdrone_predictions.pkl `
  --annotations C:\Users\xh\Desktop\WindyLab\datasets\VisDrone2019-DET-COCO\annotations\VisDrone2019-DET_val_coco.json `
  --output-dir work_dirs\video_experiments\E3_thresholds

python remdet_video\tools\analyze_temporal.py `
  --frames-jsonl work_dirs\video_experiments\tiny_640\frames.jsonl `
  --output-dir work_dirs\video_experiments\temporal

python remdet_video\tools\build_experiment_report.py
```

The generated report is under
`work_dirs/video_experiments/report/experiment_report.md`.

## Verified upstream caveat

The repository's `RepDWConv.switch_to_deploy()` does not currently pass an
equivalence smoke test. Its 3x3 branch is grouped while its 1x1 branch is not,
but the conversion creates a grouped convolution from the broadcast-summed
weights. This produces an invalid channel shape at runtime. Do not enable
`--switch-to-deploy` until the conversion is fixed and its outputs are checked
against the original graph.

MMEngine's EMAHook also swaps checkpoint fields before saving: the published
deployment/EMA weights are already in `state_dict`; `ema_state_dict` contains
the original training weights. Use the default `state_dict` when creating a
safe tensor-only checkpoint.
