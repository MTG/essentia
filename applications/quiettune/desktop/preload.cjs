const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('quiettuneDesktop', {
  preferences: ipcRenderer.sendSync('desktop:preferences'),
  savePreferences: value => ipcRenderer.send('desktop:savePreferences', value),
  state: () => ipcRenderer.invoke('desktop:state'),
  retry: () => ipcRenderer.invoke('desktop:retry'),
  logs: () => ipcRenderer.invoke('desktop:logs'),
  menu: () => ipcRenderer.invoke('desktop:menu'),
  onState: callback => {
    const listener = (_event, value) => callback(value)
    ipcRenderer.on('desktop:state', listener)
    return () => ipcRenderer.removeListener('desktop:state', listener)
  }
})
