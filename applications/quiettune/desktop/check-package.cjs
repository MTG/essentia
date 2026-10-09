const fs = require('node:fs')
const path = require('node:path')
const root = path.resolve(__dirname, '..')
const files = [
  'frontend/dist/index.html', 'desktop/icon.ico', 'LICENSE',
  '.Codex/desktop-runtime/python/python.exe', '.Codex/desktop-runtime/python/python311._pth',
  '.Codex/desktop-runtime/python/Lib/site-packages/tensorflow/__init__.py',
  '.Codex/desktop-runtime/node/node.exe', '.Codex/desktop-runtime/ffmpeg/ffmpeg.exe',
  '.Codex/desktop-runtime/ffmpeg/ffprobe.exe', '.Codex/desktop-runtime/manifest.json',
  '.Codex/desktop-runtime/licenses/Node-LICENSE.txt',
  'backend/node_modules/essentia.js/dist/essentia-wasm.umd.js',
  'backend/node_modules/essentia.js/dist/essentia.js-core.umd.js'
]
for (const name of ['msd-musicnn-1', 'mood_aggressive-msd-musicnn-1', 'mood_relaxed-msd-musicnn-1']) {
  for (const extension of ['pb', 'json']) files.push(`models/${name}.${extension}`)
}
const missing = files.filter(file => !fs.existsSync(path.join(root, file)))
if (missing.length) {
  console.error('打包资源不完整，已停止。请先构建界面、准备运行环境并下载模型：\n' + missing.join('\n'))
  process.exitCode = 1
} else console.log('完整界面、私有运行环境与模型资源检查通过。')
