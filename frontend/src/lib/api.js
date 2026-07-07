/**
 * api.js
 *
 * Thin fetch wrapper for the FastAPI backend (backend/main.py).
 * All calls use relative paths (e.g. "/api/chat"); in development,
 * vite.config.js proxies these to http://localhost:8000.
 */

const BASE_URL = '/api'

async function request(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail || detail
    } catch {
      // Response wasn't JSON — fall back to statusText.
    }
    throw new Error(detail)
  }

  return response
}

export async function sendChatMessage(message, sessionId) {
  const response = await request('/chat', {
    method: 'POST',
    body: JSON.stringify({ message, session_id: sessionId ?? null }),
  })
  return response.json()
}

export async function createSession() {
  const response = await request('/sessions', { method: 'POST' })
  return response.json()
}

export async function listSessions({ category, savedOnly } = {}) {
  const params = new URLSearchParams()
  if (category) params.set('category', category)
  if (savedOnly) params.set('saved_only', 'true')
  const query = params.toString() ? `?${params.toString()}` : ''
  const response = await request(`/sessions${query}`)
  return response.json()
}

export async function getSession(sessionId) {
  const response = await request(`/sessions/${sessionId}`)
  return response.json()
}

export async function saveSession(sessionId, saved = true) {
  const response = await request(`/sessions/${sessionId}/save?saved=${saved}`, {
    method: 'POST',
  })
  return response.json()
}

export async function deleteSession(sessionId) {
  const response = await request(`/sessions/${sessionId}`, { method: 'DELETE' })
  return response.json()
}

/**
 * Trigger a browser download for an exported table.
 * Returns nothing — this directly opens the browser's save dialog.
 */
export async function exportTable(format, tableData, filename = 'export') {
  const response = await request(`/export/${format}`, {
    method: 'POST',
    body: JSON.stringify({ table_data: tableData, filename }),
  })
  const blob = await response.blob()
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  const extension = format === 'excel' ? 'xlsx' : format
  link.download = `${filename}.${extension}`
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(url)
}
