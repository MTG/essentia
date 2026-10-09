const { app, BrowserWindow, Menu, dialog, ipcMain, screen, shell } = require('electron')
const fs = require('node:fs')
const path = require('node:path')
const { AnalysisService } = require('./service.cjs')

app.setName('QuietTune')
const home = process.env.QUIETTUNE_DESKTOP_HOME || path.join(app.getPath('appData'), 'QuietTune')
app.setPath('userData', home)
let window
let service
let starting = false
let quitting = false
let allowedToQuit = false
let state = { status: 'starting', message: '正在准备本地分析环境…' }
const windowPath = path.join(home, 'window.json')
const preferencesPath = path.join(home, 'preferences.json')
let preferences = {}
try { preferences = JSON.parse(fs.readFileSync(preferencesPath, 'utf8')) } catch { /* 首次启动采用默认值。 */ }
const logPath = path.join(home, 'logs', 'backend.log')
const root = app.isPackaged ? path.join(process.resourcesPath, 'app') : path.resolve(__dirname, '..')
const data = process.env.QUIETTUNE_DATA || (app.isPackaged ? path.join(home, 'data') : path.join(root, 'data'))
const models = process.env.QUIETTUNE_MODELS || (app.isPackaged ? path.join(home, 'models') : path.join(root, 'models'))
const titleBarHeight = 52
let titleBarTheme

function updateTitleBar() {
  if (!window || window.isDestroyed()) return
  const startup = window.webContents.getURL().startsWith('file:')
  const theme = startup ? 'startup' : preferences.theme === 'light' ? 'light' : 'dark'
  if (theme === titleBarTheme) return
  titleBarTheme = theme
  const color = theme === 'light' ? '#f4f5f8' : theme === 'startup' ? '#10141b' : '#11131b'
  window.setBackgroundColor(color)
  if (process.platform !== 'darwin') {
    window.setTitleBarOverlay({ color, symbolColor: theme === 'light' ? '#242838' : '#e5e7ef', height: titleBarHeight })
  }
}

function updateState(status, message) {
  state = { status, message }
  if (window && !window.isDestroyed()) window.webContents.send('desktop:state', state)
}

function modelCache() {
  if (!app.isPackaged || process.env.QUIETTUNE_MODELS) return
  fs.mkdirSync(models, { recursive: true })
  for (const name of fs.readdirSync(path.join(process.resourcesPath, 'models'))) {
    const destination = path.join(models, name)
    if (!fs.existsSync(destination)) fs.copyFileSync(path.join(process.resourcesPath, 'models', name), destination)
  }
}

function serviceOptions() {
  const runtime = path.join(process.resourcesPath, 'runtime')
  const environment = { QUIETTUNE_DATA: data, QUIETTUNE_MODELS: models, NUMBA_CACHE_DIR: path.join(home, 'cache', 'numba') }
  if (app.isPackaged) {
    Object.assign(environment, {
      QUIETTUNE_NODE: path.join(runtime, 'node', 'node.exe'),
      QUIETTUNE_FFMPEG: path.join(runtime, 'ffmpeg', 'ffmpeg.exe'),
      QUIETTUNE_FFPROBE: path.join(runtime, 'ffmpeg', 'ffprobe.exe'),
      QUIETTUNE_ESSENTIA_DIST: path.join(root, 'backend', 'essentia-runtime'),
      PATH: [path.join(runtime, 'python'), path.join(runtime, 'ffmpeg'), process.env.PATH || ''].join(path.delimiter)
    })
  }
  return { root, logPath, environment,
    python: app.isPackaged ? path.join(runtime, 'python', 'python.exe') : path.join(root, '.venv', 'Scripts', 'python.exe') }
}

async function showFailure(message) {
  if (quitting || !window || window.isDestroyed()) return
  updateState('error', message)
  await window.loadFile(path.join(__dirname, 'startup.html'))
}

async function startService() {
  if (starting || quitting) return
  starting = true
  try {
    if (service) await service.stop()
    updateState('starting', '正在启动本地分析服务，首次启动可能需要稍等…')
    modelCache()
    service = new AnalysisService(serviceOptions())
    service.on('failed', message => { if (!starting) showFailure(message).catch(() => {}) })
    const url = await service.start()
    if (!quitting) {
      updateState('ready', '分析环境已就绪。')
      await window.loadURL(url)
    }
  } catch (error) {
    if (service) await service.stop()
    await showFailure(error.message)
  } finally { starting = false }
}

function windowBounds() {
  const defaults = { width: 1380, height: 900 }
  try {
    const saved = JSON.parse(fs.readFileSync(windowPath, 'utf8'))
    if (!['x', 'y', 'width', 'height'].every(key => Number.isFinite(saved[key]))) return defaults
    const area = screen.getDisplayMatching(saved).workArea
    return { width: Math.min(area.width, Math.max(960, saved.width)), height: Math.min(area.height, Math.max(680, saved.height)),
      x: Math.max(area.x, Math.min(saved.x, area.x + area.width - 200)), y: Math.max(area.y, Math.min(saved.y, area.y + area.height - 100)) }
  } catch { return defaults }
}

function createWindow() {
  window = new BrowserWindow({ ...windowBounds(), minWidth: 960, minHeight: 680, title: 'QuietTune — 音乐舒适度分析',
    titleBarStyle: 'hidden',
    ...(process.platform !== 'darwin' ? { titleBarOverlay: { color: '#10141b', symbolColor: '#e5e7ef', height: titleBarHeight } } : {}),
    backgroundColor: '#10141b', icon: path.join(__dirname, 'icon.png'), show: !process.argv.includes('--quiettune-test'),
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false } })
  window.on('close', () => {
    try { if (!window.isMinimized()) fs.writeFileSync(windowPath, JSON.stringify(window.getNormalBounds()), 'utf8') }
    catch (error) { console.error('无法保存窗口位置：', error.message) }
  })
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) shell.openExternal(url)
    return { action: 'deny' }
  })
  window.webContents.on('did-finish-load', updateTitleBar)
  window.webContents.on('will-navigate', (event, url) => {
    if (service?.url && !url.startsWith(service.url + '/') && url !== service.url) {
      event.preventDefault()
      if (/^https?:\/\//.test(url)) shell.openExternal(url)
    }
  })
  window.webContents.session.on('will-download', (_event, item) => {
    item.setSaveDialogOptions({ title: '保存导出的分析结果', defaultPath: item.getFilename(),
      filters: [{ name: 'CSV 表格', extensions: ['csv'] }] })
  })
  window.loadFile(path.join(__dirname, 'startup.html')).then(startService)
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: '文件', submenu: [
      { label: '打开音乐数据目录', click: () => shell.openPath(data) },
      { label: '打开模型缓存目录', click: () => shell.openPath(models) },
      { type: 'separator' }, { label: '退出', accelerator: 'Alt+F4', click: () => app.quit() }
    ] },
    { label: '编辑', submenu: [ { label: '撤销', role: 'undo' }, { label: '重做', role: 'redo' }, { type: 'separator' },
      { label: '剪切', role: 'cut' }, { label: '复制', role: 'copy' }, { label: '粘贴', role: 'paste' }, { label: '全选', role: 'selectAll' } ] },
    { label: '视图', submenu: [ { label: '重新加载', role: 'reload' }, { label: '放大', role: 'zoomIn' },
      { label: '缩小', role: 'zoomOut' }, { label: '实际大小', role: 'resetZoom' }, { label: '全屏', role: 'togglefullscreen' } ] },
    { label: '帮助', submenu: [
      { label: '查看运行日志', click: () => shell.openPath(path.dirname(logPath)) },
      { label: '第三方许可', click: () => shell.openPath(app.isPackaged ? path.join(process.resourcesPath, 'LICENSE-NOTICES.md') : path.join(root, 'LICENSE-NOTICES.md')) },
      { label: '关于 QuietTune', click: () => dialog.showMessageBox(window, { type: 'info', title: '关于 QuietTune',
        message: `QuietTune ${app.getVersion()}`, detail: '音乐舒适度分析与个人偏好学习\n音频和评价保存在本机。\n内含的音乐模型仅限非商业用途，详见第三方许可。' }) }
    ] }
  ]))
  // 原生窗口按钮嵌入界面，菜单由页面入口弹出，快捷键继续由 Electron 管理。
  window.setMenuBarVisibility(false)
}

ipcMain.handle('desktop:state', () => state)
ipcMain.handle('desktop:retry', () => { startService(); return true })
ipcMain.handle('desktop:logs', () => shell.openPath(path.dirname(logPath)))
ipcMain.handle('desktop:menu', () => {
  if (!window || window.isDestroyed()) return false
  Menu.getApplicationMenu()?.popup({ window })
  return true
})
ipcMain.on('desktop:preferences', event => { event.returnValue = preferences })
ipcMain.on('desktop:savePreferences', (_event, value) => {
  if (!value || !['dark', 'light'].includes(value.theme) || !Number.isFinite(value.volume) || value.volume < 0 || value.volume > 1) return
  preferences = { theme: value.theme, volume: value.volume }
  updateTitleBar()
  try { fs.writeFileSync(preferencesPath, JSON.stringify(preferences), 'utf8') }
  catch (error) { console.error('无法保存界面偏好：', error.message) }
})
if (!app.requestSingleInstanceLock()) app.quit()
else {
  app.on('second-instance', () => {
    if (window) { if (window.isMinimized()) window.restore(); window.show(); window.focus() }
  })
  app.whenReady().then(() => { fs.mkdirSync(home, { recursive: true }); createWindow() }).catch(error => {
    dialog.showErrorBox('无法打开 QuietTune', `无法准备应用数据目录：${error.message}`)
    app.quit()
  })
  app.on('window-all-closed', () => app.quit())
  app.on('before-quit', event => {
    if (allowedToQuit) return
    event.preventDefault()
    if (quitting) return
    quitting = true
    updateState('starting', '正在保存并关闭分析服务…')
    Promise.resolve(service?.stop()).finally(() => { allowedToQuit = true; app.quit() })
  })
}
