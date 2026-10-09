// 真实 Electron 与 Windows 原生命中验证，所有数据位于独立项目目录。
const fs = require('node:fs')
const path = require('node:path')
const { execFileSync } = require('node:child_process')
const root = path.resolve(__dirname, '..')
const driver = execFileSync('python', ['-c', 'from pathlib import Path; import playwright; print(Path(playwright.__file__).parent / "driver" / "package")'], { encoding: 'utf8' }).trim()
const { _electron } = require(driver)
const version = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).version
const work = path.join(__dirname, 'desktop-test', `titlebar-${Date.now()}`)
const blocked = path.join(work, 'data')
fs.mkdirSync(work, { recursive: true })
fs.writeFileSync(blocked, '用于验证启动错误页的普通文件。', 'utf8')
const environment = { ...process.env, QUIETTUNE_DESKTOP_HOME: path.join(work, 'home'), QUIETTUNE_DATA: blocked }
for (const key of ['ELECTRON_RUN_AS_NODE', 'PYTHONHOME', 'PYTHONPATH', 'QUIETTUNE_MODELS', 'QUIETTUNE_NODE', 'QUIETTUNE_FFMPEG', 'QUIETTUNE_FFPROBE']) delete environment[key]
const executable = process.env.QUIETTUNE_TEST_EXE || path.join(root, '.Codex/releases/win-unpacked/QuietTune.exe')
const dev = process.argv.includes('--dev')
const errors = []
const screenshots = path.join(__dirname, 'screenshots')
fs.mkdirSync(screenshots, { recursive: true })
let windowHandle
const native = (command, handle, specification) => JSON.parse(execFileSync('python', [path.join(__dirname, 'window-native.py'), command, handle,
  typeof specification === 'object' ? JSON.stringify(specification) : specification], { encoding: 'utf8', env: { ...process.env, PYTHONUTF8: '1' } }))
function check(condition, message) { if (!condition) throw new Error(message) }
async function waitFor(predicate, message) {
  const deadline = Date.now() + 15000
  while (Date.now() < deadline) { if (await predicate()) return; await new Promise(resolve => setTimeout(resolve, 100)) }
  throw new Error(message)
}
async function inspect(page, selector, controls, screenshot) {
  const layout = await page.evaluate(({ selector, controls }) => {
    const area = navigator.windowControlsOverlay.getTitlebarAreaRect()
    const header = document.querySelector(selector)
    const bounds = header.getBoundingClientRect()
    const clickable = document.querySelector(controls)
    const button = clickable.getBoundingClientRect()
    const point = { x: button.x + button.width / 2, y: button.y + button.height / 2 }
    const right = area.x + area.width
    const gap = innerWidth - right
    return { width: innerWidth, height: innerHeight, visible: navigator.windowControlsOverlay.visible,
      area: { x: area.x, y: area.y, width: area.width, height: area.height }, button_right: button.right,
      header_top: bounds.top, header_height: bounds.height,
      drag_style: getComputedStyle(header).getPropertyValue('app-region'),
      button_style: getComputedStyle(clickable).getPropertyValue('app-region'),
      points: { drag: { x: (bounds.left + button.left) / 2, y: bounds.top + 26 }, button: point,
        minimize: { x: right + gap / 6, y: 26 }, maximize: { x: right + gap / 2, y: 26 }, close: { x: right + gap * 5 / 6, y: 26 } } }
  }, { selector, controls })
  // 启动页按钮在左侧，拖动命中选取中间空白。
  if (selector === '.titlebar') layout.points.drag.x = layout.area.width / 2
  check(layout.visible && layout.area.height === 52, '原生窗口按钮未融入 52 像素标题区。')
  check(layout.header_top === 0 && layout.header_height === 52, '标题区没有固定在窗口顶部。')
  check(layout.button_right < layout.area.x + layout.area.width, '操作按钮被系统窗口按钮遮挡。')
  check(layout.drag_style === 'drag' && layout.button_style === 'no-drag', '拖动与点击区域没有正确分离。')
  const result = native('inspect', windowHandle, { ...layout, capture: screenshot })
  check(result.hits.drag === 2 && result.hits.button === 1, `原生拖动命中错误：${JSON.stringify(result.hits)}`)
  check(result.hits.minimize === 8 && result.hits.maximize === 9 && result.hits.close === 20, `窗口按钮命中错误：${JSON.stringify(result.hits)}`)
  return { layout, native: result }
}

async function verify() {
  const desktop = await _electron.launch({ executablePath: dev ? path.join(root, 'node_modules/electron/dist/electron.exe') : executable,
    args: dev ? [root, '--quiettune-test'] : ['--quiettune-test'], cwd: root, env: environment, timeout: 60000 })
  let closing = false
  let url
  try {
    const page = await desktop.firstWindow()
    page.on('pageerror', error => errors.push(error.message))
    await page.getByRole('button', { name: '重新启动', exact: true }).waitFor({ timeout: 90000 })
    windowHandle = await desktop.evaluate(({ BrowserWindow, screen }) => {
      const win = BrowserWindow.getAllWindows()[0]
      const area = screen.getPrimaryDisplay().workArea
      win.setBounds({ x: area.x + 15, y: area.y + 15, width: Math.min(1200, area.width - 30), height: Math.min(800, area.height - 30) })
      win.show()
      win.focus()
      return win.getNativeWindowHandle().readBigUInt64LE().toString()
    })
    await waitFor(() => page.evaluate(() => navigator.windowControlsOverlay?.visible), '启动页窗口控制区没有就绪。')
    const startup = await inspect(page, '.titlebar', '#menu', path.join(screenshots, 'titlebar-startup-window.png'))
    fs.unlinkSync(blocked)
    await page.getByRole('button', { name: '重新启动', exact: true }).click()
    await page.getByRole('heading', { name: '音乐舒适度总览' }).waitFor({ timeout: 90000 })
    url = page.url()
    check(await desktop.evaluate(({ app }) => app.getVersion()) === version, '桌面版本没有更新。')
    check(!await desktop.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isMenuBarVisible()), '仍显示额外的系统菜单条。')
    await page.getByRole('button', { name: '通用评分', exact: true }).click()
    await page.getByRole('button', { name: '个人评分', exact: true }).waitFor()
    await page.getByRole('button', { name: '个人评分', exact: true }).click()
    const dark = await inspect(page, '.topbar', '[aria-label="应用菜单"]', path.join(screenshots, 'titlebar-dark-window.png'))
    await desktop.evaluate(({ Menu }) => { globalThis.menuOpened = false; Menu.getApplicationMenu().once('menu-will-show', () => { globalThis.menuOpened = true }) })
    await page.getByRole('button', { name: '应用菜单', exact: true }).click()
    check(await desktop.evaluate(() => globalThis.menuOpened), '顶部菜单按钮没有打开原生菜单。')
    const labels = await desktop.evaluate(({ Menu, BrowserWindow }) => {
      Menu.getApplicationMenu().closePopup(BrowserWindow.getAllWindows()[0])
      return Menu.getApplicationMenu().items.map(item => item.label)
    })
    check(JSON.stringify(labels) === JSON.stringify(['文件', '编辑', '视图', '帮助']), '原有中文菜单不完整。')
    await page.getByRole('button', { name: '切换浅色主题' }).click()
    await waitFor(() => desktop.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].getBackgroundColor().toLowerCase() === '#f4f5f8'), '窗口没有同步浅色主题。')
    const background = await desktop.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].getBackgroundColor())
    const light = await inspect(page, '.topbar', '[aria-label="应用菜单"]', path.join(screenshots, 'titlebar-light-window.png'))
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight))
    check(await page.locator('.topbar').evaluate(header => header.getBoundingClientRect().top) === 0, '页面滚动后无法从顶部拖动窗口。')
    await page.evaluate(() => window.scrollTo(0, 0))
    await desktop.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setBounds({ width: 960, height: 680 }))
    await page.waitForTimeout(250)
    await inspect(page, '.topbar', '[aria-label="应用菜单"]')
    await desktop.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setFullScreen(true))
    await waitFor(() => desktop.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isFullScreen()), '菜单全屏行为无效。')
    await page.waitForTimeout(250)
    const fullscreen = await page.locator('.topbar').evaluate(header => {
      const overlay = navigator.windowControlsOverlay
      const area = overlay.getTitlebarAreaRect()
      return { padding: getComputedStyle(header).paddingRight, controls_visible: overlay.visible,
        available_right: overlay.visible ? area.x + area.width : innerWidth,
        actions: document.querySelector('.topbar-actions').getBoundingClientRect().toJSON(), width: innerWidth }
    })
    check(fullscreen.actions.width > 0 && fullscreen.actions.left >= 0 && fullscreen.actions.right <= fullscreen.available_right,
      `全屏后顶部操作区布局异常：${JSON.stringify(fullscreen)}`)
    await desktop.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setFullScreen(false))
    await waitFor(() => desktop.evaluate(({ BrowserWindow }) => !BrowserWindow.getAllWindows()[0].isFullScreen()), '退出全屏失败。')
    await page.waitForTimeout(250)
    native('system', windowHandle, 'maximize')
    await waitFor(() => desktop.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isMaximized()), '原生最大化命令无效。')
    await inspect(page, '.topbar', '[aria-label="应用菜单"]')
    native('system', windowHandle, 'restore')
    await waitFor(() => desktop.evaluate(({ BrowserWindow }) => !BrowserWindow.getAllWindows()[0].isMaximized()), '原生还原命令无效。')
    native('system', windowHandle, 'minimize')
    await waitFor(() => desktop.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isMinimized()), '原生最小化命令无效。')
    native('system', windowHandle, 'restore')
    await waitFor(() => desktop.evaluate(({ BrowserWindow }) => !BrowserWindow.getAllWindows()[0].isMinimized()), '窗口未从最小化恢复。')
    const exited = new Promise(resolve => desktop.process().once('exit', resolve))
    closing = true
    native('system', windowHandle, 'close')
    let closeTimeout
    try { await Promise.race([exited, new Promise((_, reject) => { closeTimeout = setTimeout(() => reject(new Error('原生关闭后应用未退出。')), 30000) })]) }
    finally { clearTimeout(closeTimeout) }
    let stopped = false
    try { await fetch(new URL('/api/health', url), { signal: AbortSignal.timeout(1500) }) }
    catch (error) { stopped = error.cause?.code === 'ECONNREFUSED' || error.cause?.errors?.every(item => item.code === 'ECONNREFUSED') === true }
    check(stopped, '关闭窗口后分析服务未清理。')
    check(!errors.length, `页面存在错误：${errors.join('；')}`)
    fs.writeFileSync(path.join(__dirname, 'titlebar-result.json'), JSON.stringify({ status: '通过', version, executable: dev ? '开发版 Electron' : executable,
      startup, dark, light, background, labels, fullscreen, console_errors: errors, checked: ['启动与错误页', '原生拖动/按钮命中', '原生控制区无重叠', '顶部菜单', '深浅主题', '滚动保持顶部', '最小宽度', '全屏和退出全屏', '最大化还原', '最小化恢复', '原生关闭与服务清理'] }, null, 2), 'utf8')
    console.log('标题栏、原生窗口控制、主题、菜单和关闭清理全部通过。')
  } finally { if (!closing && desktop.process().exitCode === null) await desktop.close() }
}
verify().catch(error => { console.error('窗口顶部验证失败：', error.message); process.exitCode = 1 })
