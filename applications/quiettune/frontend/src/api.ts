export async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch('/api' + url, init)
  if (!response.ok) {
    const data = await response.json().catch(() => ({ detail: '服务器未返回有效响应。' }))
    const detail = typeof data.detail === 'string' ? data.detail : '提交的内容不符合要求，请检查输入。'
    throw new Error(detail)
  }
  return response.json()
}
export function send<T>(url: string, data?: unknown, method = 'POST') {
  return api<T>(url, { method, headers: { 'Content-Type': 'application/json' }, body: data === undefined ? undefined : JSON.stringify(data) })
}
export function time(seconds: number) {
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`
}
export function number(value: number | null | undefined, digits = 1) {
  return value == null ? '—' : value.toFixed(digits)
}
