interface Window {
  quiettuneDesktop?: {
    preferences: { theme?: 'dark' | 'light'; volume?: number }
    savePreferences: (value: { theme: string; volume: number }) => void
    state: () => Promise<{ status: string; message: string }>
    retry: () => Promise<boolean>
    logs: () => Promise<string>
    menu: () => Promise<boolean>
    onState: (callback: (value: { status: string; message: string }) => void) => () => void
  }
}
