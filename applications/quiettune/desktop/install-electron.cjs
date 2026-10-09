// 使用 Electron 官方下载器及其标准代理支持，沿用用户提供的网络配置。
if (process.env.HTTP_PROXY || process.env.HTTPS_PROXY) process.env.ELECTRON_GET_USE_PROXY = 'true'
require('electron/install.js')
