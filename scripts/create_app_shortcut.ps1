$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$iconPath = Join-Path $projectRoot 'assets/app-icon/playmodel.ico'
if (-not (Test-Path -LiteralPath $iconPath)) { throw 'Build the app icon first.' }
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $projectRoot 'PlayModel.lnk'))
$shortcut.TargetPath = Join-Path $env:WINDIR 'System32/wscript.exe'
$shortcut.Arguments = '"' + (Join-Path $projectRoot 'PlayModel.vbs') + '"'
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = $iconPath + ',0'
$shortcut.Description = 'PlayModel local learning'
$shortcut.Save()
