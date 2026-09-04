# First-appearance experiment

This module evaluates a mission-level condition that ordinary image mAP does
not measure: a target class must not trigger before its first annotated
appearance and must be detected on that first frame.

Run all commands from the RemDet project root in the `remdet5080` Conda
environment.

## 1. Build the event manifest

```powershell
python remdet_video\tools\build_first_appearance_manifest.py
```

The default dataset root is
`C:\Users\xh\Desktop\WindyLab\datasets`. Use `--dataset-root` to override it.

## 2. Cache RemDet-S predictions for the van experiment

```powershell
python remdet_video\tools\run_first_appearance_inference.py
```

The command uses the same sanitized RemDet-S checkpoint that was exported for
Jetson. Predictions are cached at a low score threshold so threshold sweeps do
not rerun the neural network.

## 3. Evaluate van onset events

```powershell
python remdet_video\tools\evaluate_first_appearance.py `
  --output-dir work_dirs\first_appearance\evaluation\remdet_s_fp32_van `
  --thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.34 0.35 0.36 0.37 0.38 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.92 0.94 0.96 0.98 0.99
```

## 4. Reproduce the all-class held-out experiment

```powershell
python remdet_video\tools\run_first_appearance_inference.py `
  --events work_dirs\first_appearance\manifest\candidate_events.json `
  --splits val test-dev `
  --output-dir work_dirs\first_appearance\inference\remdet_s_fp32_all_classes_eval

python remdet_video\tools\evaluate_first_appearance.py `
  --events work_dirs\first_appearance\manifest\candidate_events.json `
  --inference-dir work_dirs\first_appearance\inference\remdet_s_fp32_all_classes_eval `
  --output-dir work_dirs\first_appearance\evaluation\remdet_s_fp32_all_classes `
  --development-split val `
  --test-split test-dev `
  --thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.34 0.35 0.36 0.37 0.38 0.40 0.50 0.60 0.70 0.80 0.85 0.90 0.92 0.94 0.96 0.98 0.99
```

Add `--single-label` to the inference command to reproduce the single-label
post-processing ablation. Do not compare its end-to-end decode time with a
previous run unless the filesystem cache and benchmark order are controlled.

The consolidated Chinese report is written to
`work_dirs\first_appearance\FIRST_APPEARANCE_REPORT.md`.
