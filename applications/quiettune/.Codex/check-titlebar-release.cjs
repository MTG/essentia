// 使用现有 ASAR 工具逐字节核对发行目录，防止验证源码时遗漏旧包资源。
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const crypto = require('node:crypto')
const asar = require('@electron/asar')
const root = path.resolve(__dirname, '..')
const packaged = path.join(__dirname, 'releases/win-unpacked/resources')
const archive = path.join(packaged, 'app.asar')
const version = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version
const digests = {}
function record(relative, expected, actual) {
  assert.ok(expected.equals(actual), `发行资源与源码不一致：${relative}`)
  digests[relative] = crypto.createHash('sha256').update(expected).digest('hex')
}
for (const relative of ['desktop/main.cjs', 'desktop/preload.cjs', 'desktop/service.cjs', 'desktop/startup.html']) {
  record(relative, fs.readFileSync(path.join(root, relative)), asar.extractFile(archive, relative))
}
assert.equal(JSON.parse(asar.extractFile(archive, 'package.json').toString('utf8')).version, version, '发行目录版本没有更新。')
for (const directory of ['frontend/dist', 'backend/app']) {
  for (const entry of fs.readdirSync(path.join(root, directory), { recursive: true, withFileTypes: true })) {
    if (!entry.isFile() || entry.parentPath.includes('__pycache__')) continue
    const original = path.join(entry.parentPath, entry.name)
    const relative = path.relative(root, original)
    record(relative, fs.readFileSync(original), fs.readFileSync(path.join(packaged, 'app', relative)))
  }
}
const result = { status: '通过', version, resources: digests, verified_at: new Date().toISOString() }
fs.writeFileSync(path.join(__dirname, 'titlebar-package-result.json'), JSON.stringify(result, null, 2), 'utf8')
console.log(`正式桌面版本 ${version}，共 ${Object.keys(digests).length} 个资源与源码逐字节一致。`)
