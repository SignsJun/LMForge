$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Push-Location $repoRoot
try {
    if (git status --porcelain) {
        throw "Local changes detected. Commit or stash them before pulling."
    }

    git fetch origin main
    if ($LASTEXITCODE -ne 0) {
        throw "git fetch failed"
    }

    git rebase origin/main
    if ($LASTEXITCODE -ne 0) {
        throw "git rebase failed"
    }

    Write-Output "Code is synchronized."
} finally {
    Pop-Location
}
