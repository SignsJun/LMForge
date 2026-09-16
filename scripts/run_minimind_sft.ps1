param(
    [ValidateSet("lora", "full")]
    [string]$Mode = "lora",
    [string]$Python = "python",
    [int]$MaxSteps = 0,
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$trainConfig = "configs/train/minimind_sft_$Mode.yaml"
$runArgs = @(
    "scripts/sft.py",
    "--model", "configs/model/minimind_tiny.yaml",
    "--data", "configs/data/minimind_sft.yaml",
    "--train", $trainConfig,
    "--require-cuda"
)

if ($MaxSteps -gt 0) {
    $runArgs += @("--max-steps", $MaxSteps)
}
if ($Resume) {
    $runArgs += @("--resume", $Resume)
}

Push-Location $repoRoot
try {
    & $Python @runArgs
    if ($LASTEXITCODE -ne 0) {
        throw "SFT training failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
