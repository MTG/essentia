param([switch]$Refresh, [string]$Proxy = $env:HTTPS_PROXY)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$projectRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
Set-Location -LiteralPath $projectRoot
$runtimeRoot = Join-Path $projectRoot '.Codex/desktop-runtime'
$downloadRoot = Join-Path $projectRoot '.Codex/desktop-downloads'
$pythonRoot = Join-Path $runtimeRoot 'python'
$sourcePackages = Join-Path $projectRoot '.venv/Lib/site-packages'
$sourcePython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $sourcePython)) { throw '缺少已安装依赖的项目虚拟环境，请先运行 ./start.ps1 -Install。' }
New-Item -ItemType Directory -Force $downloadRoot, $pythonRoot | Out-Null
$archive = Join-Path $downloadRoot 'python-3.11.6-embed-amd64.zip'
$sourceUrl = 'https://www.python.org/ftp/python/3.11.6/python-3.11.6-embed-amd64.zip'
if ($Refresh -or -not (Test-Path $archive)) {
    if ($Proxy) { Invoke-WebRequest -Uri $sourceUrl -Proxy $Proxy -OutFile $archive }
    else { Invoke-WebRequest -Uri $sourceUrl -OutFile $archive }
}
Expand-Archive -LiteralPath $archive -DestinationPath $pythonRoot -Force

# 嵌入式发行版只使用随应用提供的模块，不依赖系统 Python 路径。
$modulePaths = "python311.zip`n.`nLib/site-packages`n../../app`nimport site`n"
[IO.File]::WriteAllText((Join-Path $pythonRoot 'python311._pth'), $modulePaths, [Text.UTF8Encoding]::new($false))
$targetPackages = Join-Path $pythonRoot 'Lib/site-packages'
New-Item -ItemType Directory -Force $targetPackages | Out-Null
Write-Host '复制锁定的 Python 运行依赖…'
& robocopy $sourcePackages $targetPackages /E /XD __pycache__ /XF '*.pyc' /NFL /NDL /NJH /NJS /NP
if ($LASTEXITCODE -gt 7) { throw 'Python 依赖复制失败。' }

$binarySources = @{
    'node/node.exe' = (Get-Command node.exe).Source
    'ffmpeg/ffmpeg.exe' = (Get-Command ffmpeg.exe).Source
    'ffmpeg/ffprobe.exe' = (Get-Command ffprobe.exe).Source
}
foreach ($entry in $binarySources.GetEnumerator()) {
    $destination = Join-Path $runtimeRoot $entry.Key
    New-Item -ItemType Directory -Force (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $entry.Value -Destination $destination -Force
}

# 这些 Microsoft 可再发行组件随私有解释器放置，避免依赖全局安装状态。
foreach ($dll in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'concrt140.dll')) {
    $source = Join-Path $env:WINDIR "System32/$dll"
    if (Test-Path $source) { Copy-Item -LiteralPath $source -Destination $pythonRoot -Force }
}
$licenses = Join-Path $runtimeRoot 'licenses'
New-Item -ItemType Directory -Force $licenses | Out-Null
Copy-Item -LiteralPath backend/requirements-windows.lock -Destination $licenses -Force
Copy-Item -LiteralPath LICENSE-NOTICES.md -Destination $licenses -Force
Copy-Item -LiteralPath LICENSE -Destination (Join-Path $licenses 'QuietTune-LICENSE.txt') -Force
$frontendLicenses = Join-Path $licenses 'frontend'
Get-ChildItem -LiteralPath (Join-Path $projectRoot 'frontend/node_modules') -Recurse -File |
    Where-Object { $_.Name -match '^(LICENSE|LICENCE|COPYING|NOTICE)(\..*)?$' } | ForEach-Object {
        $relative = $_.FullName.Substring((Join-Path $projectRoot 'frontend/node_modules').Length + 1)
        $destination = Join-Path $frontendLicenses $relative
        New-Item -ItemType Directory -Force (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
    }
$nodeLicense = Join-Path $licenses 'Node-LICENSE.txt'
if ($Refresh -or -not (Test-Path $nodeLicense)) {
    $nodeVersion = (& node --version).Trim()
    $nodeLicenseUrl = "https://raw.githubusercontent.com/nodejs/node/$nodeVersion/LICENSE"
    if ($Proxy) { Invoke-WebRequest -Uri $nodeLicenseUrl -Proxy $Proxy -OutFile $nodeLicense }
    else { Invoke-WebRequest -Uri $nodeLicenseUrl -OutFile $nodeLicense }
}
$ffmpegRoot = Split-Path -Parent (Split-Path -Parent $binarySources['ffmpeg/ffmpeg.exe'])
foreach ($filename in @('LICENSE', 'README.txt')) {
    $source = Join-Path $ffmpegRoot $filename
    if (Test-Path $source) { Copy-Item -LiteralPath $source -Destination (Join-Path $licenses "FFmpeg-$filename") -Force }
}
$entries = @(@{ name = 'Python'; version = '3.11.6'; source = $sourceUrl; sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() })
foreach ($entry in $binarySources.GetEnumerator()) {
    $entries += @{ name = $entry.Key; sha256 = (Get-FileHash -LiteralPath $entry.Value -Algorithm SHA256).Hash.ToLower() }
}
$manifest = @{ generated_at = (Get-Date).ToString('o'); platform = 'Windows x64'; dependencies = 'licenses/requirements-windows.lock'; files = $entries }
[IO.File]::WriteAllText((Join-Path $runtimeRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
Write-Host '验证私有 Python 与真实模型依赖…'
& (Join-Path $pythonRoot 'python.exe') -c "import tensorflow, librosa, fastapi, sqlalchemy, scipy, soundfile; print('私有运行环境已就绪：TensorFlow', tensorflow.__version__)"
if ($LASTEXITCODE -ne 0) { throw '私有运行环境验证失败，停止打包。' }
Write-Host '桌面运行环境准备完成。'
