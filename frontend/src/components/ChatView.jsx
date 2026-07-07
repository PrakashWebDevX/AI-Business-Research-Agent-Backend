import { useEffect, useRef, useState } from 'react'
import { useLocation, useParams } from 'react-router-dom'
import { Send, Loader2 } from 'lucide-react'
import { getSession, sendChatMessage } from '../lib/api'
import ChatMessage from './ChatMessage'

export default function ChatView() {
  const { sessionId } = useParams()
  const location = useLocation()
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(true)
  const bottomRef = useRef(null)
  const hasSentPendingRef = useRef(false)

  useEffect(() => {
    hasSentPendingRef.current = false
    loadSession()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, sending])

  async function loadSession() {
    setLoadingHistory(true)
    try {
      const session = await getSession(sessionId)
      setMessages(session.messages)

      // If we navigated here from HomeView with a pending first
      // question, send it now that the (empty) session has loaded.
      const pending = location.state?.pendingMessage
      if (pending && !hasSentPendingRef.current) {
        hasSentPendingRef.current = true
        await submitMessage(pending)
      }
    } catch (err) {
      console.error('Failed to load session:', err)
    } finally {
      setLoadingHistory(false)
    }
  }

  async function submitMessage(text) {
    const trimmed = text.trim()
    if (!trimmed || sending) return

    setSending(true)
    setMessages((prev) => [
      ...prev,
      { id: `local-${Date.now()}`, role: 'user', content: trimmed, created_at: new Date().toISOString() },
    ])
    setInput('')

    try {
      const { message } = await sendChatMessage(trimmed, sessionId)
      setMessages((prev) => [...prev, message])
    } catch (err) {
      console.error('Chat request failed:', err)
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `Sorry, something went wrong: ${err.message}`,
          created_at: new Date().toISOString(),
        },
      ])
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {loadingHistory ? (
          <div className="flex items-center justify-center h-full text-muted text-sm gap-2">
            <Loader2 size={16} className="animate-spin" />
            Loading conversation...
          </div>
        ) : (
          <div className="max-w-3xl mx-auto">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}

            {sending && (
              <div className="flex items-center gap-2 text-muted text-xs mb-4 pl-9">
                <Loader2 size={13} className="animate-spin" />
                Thinking...
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="border-t border-border px-6 py-4">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            submitMessage(input)
          }}
          className="max-w-3xl mx-auto flex items-center gap-2 bg-card border border-border rounded-2xl px-4 py-3
                     focus-within:border-primary transition-colors"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a follow-up question..."
            className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted outline-none"
          />
          <button
            type="submit"
            disabled={sending || !input.trim()}
            className="p-2 rounded-xl bg-primary text-white disabled:opacity-40 hover:bg-primary/90 transition-colors"
          >
            <Send size={16} />
          </button>
        </form>
      </div>
    </div>
  )
}
