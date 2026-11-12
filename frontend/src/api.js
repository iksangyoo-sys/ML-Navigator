// 서버 API 호출 도우미
export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function handle(res) {
  let body = null
  try {
    body = await res.json()
  } catch {
    throw new ApiError(`서버 응답 오류 (${res.status})`, res.status)
  }
  if (!body.ok) throw new ApiError(body.error || '요청 실패', res.status)
  return body
}

export const api = {
  async get(url) {
    return (await handle(await fetch('/api' + url, { credentials: 'same-origin' }))).data
  },
  async raw(method, url, data) {
    return handle(
      await fetch('/api' + url, {
        method,
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: data === undefined ? undefined : JSON.stringify(data),
      }),
    )
  },
  async post(url, data) {
    return (await this.raw('POST', url, data ?? {})).data
  },
  async put(url, data) {
    return (await this.raw('PUT', url, data ?? {})).data
  },
  async patch(url, data) {
    return (await this.raw('PATCH', url, data ?? {})).data
  },
  async del(url) {
    return (await this.raw('DELETE', url)).data
  },
  async upload(url, files, field = 'files', extra = {}) {
    const fd = new FormData()
    for (const f of files) fd.append(field, f)
    Object.entries(extra).forEach(([k, v]) => v != null && v !== '' && fd.append(k, v))
    return handle(await fetch('/api' + url, { method: 'POST', body: fd, credentials: 'same-origin' }))
  },
}

export function downloadText(filename, text, mime = 'text/csv;charset=utf-8') {
  const blob = new Blob(['﻿' + text], { type: mime })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 2000)
}

export function tableToCsv(columns, rows) {
  const esc = (v) => {
    if (v == null) return ''
    const s = String(v)
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s
  }
  return [columns.map(esc).join(','), ...rows.map((r) => r.map(esc).join(','))].join('\n')
}

export function fmt(v, digits = 4) {
  if (v == null || v === '') return ''
  if (typeof v === 'boolean') return v ? '예' : '아니오'
  if (typeof v === 'number') {
    if (!isFinite(v)) return '-'
    if (Number.isInteger(v)) return v.toLocaleString('ko-KR')
    const a = Math.abs(v)
    if (a !== 0 && (a < 1e-3 || a >= 1e7)) return v.toExponential(3)
    if (a >= 1000) return v.toLocaleString('ko-KR', { maximumFractionDigits: 2 })
    return parseFloat(v.toPrecision(digits + (a >= 1 ? 1 : 0))).toString()
  }
  return String(v)
}
