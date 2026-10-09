param([switch]$Install, [switch]$Web, [int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location -LiteralPath $projectRoot
$env:PYTHONUTF8 = '1'

function Check-ExitCode {
    if ($LASTEXITCODE -ne 0) { throw '执行失败，已停止启动。请查看上面的错误信息。' }
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw '未找到 FFmpeg，请先安装并加入 PATH，步骤见 README.md。' }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw '未找到 Node.js，请安装 Node.js 22。' }
if (-not (Test-Path '.venv/Scripts/python.exe')) { python -m venv .venv; Check-ExitCode; $Install = $true }
if ($Install) {
    Write-Host '安装项目本地 Python 依赖…'
    & ./.venv/Scripts/python.exe -m pip install -r backend/requirements-windows.lock
    Check-ExitCode
}
if ($Install -or -not (Test-Path 'backend/node_modules/essentia.js')) { npm.cmd ci --prefix backend; Check-ExitCode }
if ($Install -or -not (Test-Path 'frontend/node_modules')) { npm.cmd ci --prefix frontend; Check-ExitCode }
if ($Install -or -not (Test-Path 'frontend/dist/index.html')) { npm.cmd run build --prefix frontend; Check-ExitCode }
$modelDirectory = if ($env:QUIETTUNE_MODELS) { $env:QUIETTUNE_MODELS } else { Join-Path $projectRoot 'models' }
$needed = @('msd-musicnn-1', 'mood_aggressive-msd-musicnn-1', 'mood_relaxed-msd-musicnn-1')
if ($needed | Where-Object { -not (Test-Path (Join-Path $modelDirectory "$_.pb")) -or -not (Test-Path (Join-Path $modelDirectory "$_.json")) }) {
    Write-Host '下载官方非商业模型，具体许可及官网声明差异见 LICENSE-NOTICES.md…'
    & ./.venv/Scripts/python.exe -m backend.download_models
    Check-ExitCode
}
if ($Web) {
    Write-Host "QuietTune 本地地址：http://127.0.0.1:$Port"
    & ./.venv/Scripts/python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port
} else {
    if ($Install -or -not (Test-Path 'node_modules/electron/dist/electron.exe')) { npm.cmd ci; Check-ExitCode }
    Write-Host '正在打开 QuietTune 桌面应用…'
    npm.cmd start
}
Check-ExitCode
