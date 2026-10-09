const { spawn } = require('node:child_process')
const { EventEmitter } = require('node:events')
const { createWriteStream, mkdirSync } = require('node:fs')
const path = require('node:path')

class AnalysisService extends EventEmitter {
  constructor(options) {
    super()
    this.options = options
    this.child = null
    this.stopping = false
    this.url = null
  }

  async start() {
    this.stopping = false
    this.url = null
    const { root, python, environment, logPath } = this.options
    mkdirSync(path.dirname(logPath), { recursive: true })
    const log = createWriteStream(logPath, { flags: 'a' })
    log.write(`\n[${new Date().toISOString()}] 正在启动分析服务\n`)
    const child = spawn(python, ['-u', '-m', 'backend.desktop_server'], {
      cwd: root, env: { ...process.env, ...environment, PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1', TF_CPP_MIN_LOG_LEVEL: '2' },
      windowsHide: true, stdio: ['pipe', 'pipe', 'pipe']
    })
    this.child = child
    let buffered = ''
    let failure = null
    child.on('error', error => { failure = error })
    child.stdout.on('data', chunk => {
      log.write(chunk)
      buffered += chunk.toString('utf8')
      let newline
      while ((newline = buffered.indexOf('\n')) >= 0) {
        const line = buffered.slice(0, newline)
        buffered = buffered.slice(newline + 1)
        try {
          const message = JSON.parse(line)
          if (message.type === 'quiettune-service' && Number.isInteger(message.port)) {
            this.url = `http://127.0.0.1:${message.port}`
          }
        } catch { /* 第三方库也可能向标准输出写普通日志。 */ }
      }
    })
    child.stderr.on('data', chunk => log.write(chunk))
    child.stdin.on('error', () => {})
    child.once('close', (code, signal) => {
      log.end(`分析服务已退出：${code ?? signal}\n`)
      if (this.child === child) this.child = null
      if (!this.stopping) this.emit('failed', `分析服务意外退出（${code ?? signal}）。请查看日志后重试。`)
    })
    const deadline = Date.now() + 120000
    while (Date.now() < deadline) {
      if (this.stopping) throw new Error('应用正在关闭。')
      if (failure) throw new Error(`无法启动分析服务：${failure.message}`)
      if (child.exitCode !== null || child.signalCode !== null) throw new Error('分析服务启动失败，请查看运行日志。')
      if (this.url) {
        try {
          const response = await fetch(this.url + '/api/health', { signal: AbortSignal.timeout(1200) })
          if (response.ok && (await response.json()).status === 'ok') return this.url
        } catch { /* 首次导入分析库需要时间，继续等待健康响应。 */ }
      }
      await new Promise(resolve => setTimeout(resolve, 200))
    }
    throw new Error('分析服务启动超过两分钟，请查看日志并重试。')
  }

  async stop() {
    this.stopping = true
    const child = this.child
    if (!child) return
    const closed = new Promise(resolve => child.once('close', resolve))
    if (child.stdin.writable) child.stdin.end('shutdown\n')
    const finished = await Promise.race([closed.then(() => true), new Promise(resolve => {
      const timer = setTimeout(() => resolve(false), 12000)
      timer.unref()
    })])
    if (!finished && child.exitCode === null) {
      // 仅结束本应用创建的进程树；未完成任务由数据库在下次启动恢复。
      if (process.platform === 'win32') {
        await new Promise(resolve => {
          const killer = spawn('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true })
          killer.once('close', resolve)
          killer.once('error', resolve)
        })
      } else child.kill('SIGKILL')
      await closed
    }
  }
}

module.exports = { AnalysisService }
