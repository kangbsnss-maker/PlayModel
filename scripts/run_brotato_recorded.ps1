param(
    [int]$Waves = 2,
    [int]$Seconds = 180,
    [string]$Style = 'configs/styles/balanced.json'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
& 'C:/Users/1/.codex/tools/Invoke-Hidden.ps1' `
    -FilePath (Join-Path $ProjectRoot '.venv/Scripts/python.exe') `
    -ArgumentList @('scripts/run_brotato_session.py', '--record', '--waves', "$Waves", '--seconds', "$Seconds", '--style', $Style) `
    -WorkingDirectory $ProjectRoot -TimeoutSeconds ($Seconds + 600)
exit $LASTEXITCODE
