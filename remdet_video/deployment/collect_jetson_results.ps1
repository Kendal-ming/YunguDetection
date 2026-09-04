param(
    [string]$RemoteHost = 'nvidia@192.168.55.1',
    [string]$CondaEnvironment = 'remdet5080'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$DeploymentDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Resolve-Path (Join-Path $DeploymentDirectory '..\..')
$ResultsDirectory = Join-Path $RepositoryRoot 'work_dirs\deployment\jetson_results'
$ArchivePath = Join-Path $ResultsDirectory 'remdet_all_results.tar.gz'
$EvaluationScript = Join-Path $DeploymentDirectory 'evaluate_visdrone_predictions.py'
$SummaryScript = Join-Path $DeploymentDirectory 'summarize_jetson_results.py'
$PredictionsPath = Join-Path $ResultsDirectory 'visdrone_val_predictions_trt_fp16.json'
$EvaluationPath = Join-Path $ResultsDirectory 'visdrone_val_coco_eval.json'

New-Item -ItemType Directory -Force -Path $ResultsDirectory | Out-Null

Write-Host 'Downloading the complete Jetson result bundle...'
& scp "${RemoteHost}:/home/nvidia/remdet_deploy/remdet_all_results.tar.gz" $ArchivePath
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
}

Write-Host 'Extracting results...'
& tar -xzf $ArchivePath -C $ResultsDirectory
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $PredictionsPath -PathType Leaf)) {
    throw "Prediction file was not found after extraction: $PredictionsPath"
}

Write-Host 'Evaluating TensorRT FP16 predictions against VisDrone ground truth...'
& conda run -n $CondaEnvironment python $EvaluationScript `
    --predictions $PredictionsPath `
    --output $EvaluationPath
$EvaluationExitCode = $LASTEXITCODE

Write-Host 'Generating final JSON, CSV, Markdown report and comparison chart...'
& conda run -n $CondaEnvironment python $SummaryScript `
    --results-dir $ResultsDirectory
$SummaryExitCode = $LASTEXITCODE

Write-Host ''
Write-Host "Extracted results: $ResultsDirectory"
Write-Host "COCO evaluation report: $EvaluationPath"
Write-Host "Final report: $(Join-Path $ResultsDirectory 'jetson_deployment_report.md')"
Write-Host "Comparison chart: $(Join-Path $ResultsDirectory 'jetson_deployment_comparison.png')"
if ($EvaluationExitCode -ne 0) {
    throw "Evaluation completed but exceeded the baseline tolerance (exit code $EvaluationExitCode). Inspect the report."
}
if ($SummaryExitCode -ne 0) {
    throw "Final result summarization failed with exit code $SummaryExitCode."
}
Write-Host 'Jetson inference and Windows COCO evaluation both passed.'
