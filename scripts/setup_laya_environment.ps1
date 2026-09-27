param()
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Hidden = 'C:/Users/1/.codex/tools/Invoke-Hidden.ps1'
$BasePython = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
$TargetEnvironment = Join-Path $ProjectRoot '.venv-laya'
& $Hidden -FilePath $BasePython -ArgumentList @('-m', 'venv', '--system-site-packages', $TargetEnvironment) -TimeoutSeconds 120
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$BasePackages = Join-Path $ProjectRoot '.venv/Lib/site-packages'
Set-Content -LiteralPath (Join-Path $TargetEnvironment 'Lib/site-packages/playmodel_base.pth') -Value $BasePackages -Encoding ASCII
$LayaPython = Join-Path $TargetEnvironment 'Scripts/python.exe'
& $Hidden -FilePath $LayaPython -ArgumentList @('-m', 'pip', 'install', '-r', (Join-Path $ProjectRoot 'configs/requirements-laya.txt')) -TimeoutSeconds 600
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Hidden -FilePath $LayaPython -ArgumentList @('-X', 'utf8', (Join-Path $PSScriptRoot 'setup_laya.py')) -TimeoutSeconds 900
exit $LASTEXITCODE
