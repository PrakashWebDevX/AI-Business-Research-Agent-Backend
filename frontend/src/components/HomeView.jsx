import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Send, Sparkles } from 'lucide-react'
import { createSession } from '../lib/api'

const EXAMPLE_QUESTIONS = [
  'Show highest paid employees',
  'Top selling products',
  'Average salary',
  'Research NVIDIA',
  'Latest AI News',
  'Explain LangGraph',
]

export default function HomeView() {
  const navigate = useNavigate()
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)

  async function handleAsk(question) {
    const text = (question ?? input).trim()
    if (!text || sending) return

    setSending(true)
    try {
      const { session_id } = await createSession()
      // Navigate immediately so the chat view takes over and shows a
      // loading state while the first answer streams in.
      navigate(`/chat/${session_id}`, { state: { pendingMessage: text } })
    } catch (err) {
      console.error('Failed to start chat:', err)
      setSending(false)
    }
  }

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6">
      <div className="w-full max-w-2xl text-center">
        <div className="flex items-center justify-center gap-2 mb-3">
          <Sparkles size={28} className="text-primary" />
        </div>
        <h2 className="text-2xl font-semibold text-foreground mb-1">Ask Anything...</h2>
        <p className="text-sm text-muted mb-8">
          Internal business data or live web research — I'll figure out which to use.
        </p>

        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleAsk()
          }}
          className="flex items-center gap-2 bg-card border border-border rounded-2xl px-4 py-3 mb-8
                     focus-within:border-primary transition-colors"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your business or the web..."
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

        <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-3">
          Example Questions
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {EXAMPLE_QUESTIONS.map((question) => (
            <button
              key={question}
              onClick={() => handleAsk(question)}
              disabled={sending}
              className="px-3 py-2.5 rounded-xl text-xs text-left text-foreground
                         bg-card border border-border hover:border-primary hover:text-primary
                         transition-colors disabled:opacity-40"
            >
              {question}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
