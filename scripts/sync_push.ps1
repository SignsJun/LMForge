param(
    [string]$Message = "Sync LMForge code and model"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Push-Location $repoRoot
try {
    git add -A
    if ($LASTEXITCODE -ne 0) {
        throw "git add failed"
    }

    if (git status --porcelain) {
        git commit -m $Message
        if ($LASTEXITCODE -ne 0) {
            throw "git commit failed"
        }
    }

    git pull --rebase origin main
    if ($LASTEXITCODE -ne 0) {
        throw "git pull --rebase failed; resolve the conflict before pushing"
    }

    git push origin main
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed"
    }
} finally {
    Pop-Location
}
