param(
    [string]$Image = "openmythos-distill:local",
    [string]$Data = "/workspace/data/openmythos_scaled_refinement_dataset.jsonl",
    [string]$ModelSize = "medium",
    [int]$Steps = 1200,
    [int]$SeqLen = 384,
    [int]$TrainLoop = 4,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 8,
    [int]$MaxEvalPerSplit = 160,
    [string]$ScoresOut = "/workspace/data/openmythos_loop_scores.jsonl",
    [string]$ReportOut = "/workspace/data/reports/openmythos_ce_export_report.json",
    [string]$ArtifactDir = "/workspace/artifacts/openmythos-scaled-ce-export"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path ".").Path

docker run --rm --gpus all `
    -v "${repo}:/workspace" `
    --entrypoint python `
    $Image `
    /workspace/scripts/export_openmythos_ce.py `
    --data $Data `
    --require-cuda `
    --amp `
    --train-stages `
    --model-size $ModelSize `
    --steps $Steps `
    --seq-len $SeqLen `
    --train-loop $TrainLoop `
    --batch-size $BatchSize `
    --grad-accum-steps $GradAccumSteps `
    --max-eval-per-split $MaxEvalPerSplit `
    --eval-loops 1 2 4 `
    --artifact-dir $ArtifactDir `
    --scores-out $ScoresOut `
    --report-out $ReportOut
