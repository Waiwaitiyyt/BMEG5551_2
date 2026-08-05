$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = (Resolve-Path "$ScriptDir/..").Path

Set-Location $ProjectDir
$env:NLTK_DISABLE_IMPORT_SECURITY = "1"
uv run label-studio start
