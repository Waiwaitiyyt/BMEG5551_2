$ErrorActionPreference = "Stop"

$installed = uv pip show label-studio 2>$null
if (-not $?) {
    Write-Host "Error: Current uv environment has not installed label-studio" -ForegroundColor Red
    Write-Host "Please run the following command to install:" -ForegroundColor Yellow
    Write-Host "  uv pip install label-studio"
    Write-Host "Or if using uv project management:" -ForegroundColor Yellow
    Write-Host "  uv add label-studio"
    exit 1
}

uv run label-studio start