import { useEffect, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  Plus,
  MessageSquare,
  Database,
  Globe,
  Bookmark,
  Settings,
  LogOut,
  Sparkles,
} from 'lucide-react'
import { createSession, listSessions } from '../lib/api'

const NAV_SECTIONS = [
  { key: 'chat', label: 'Chat History', icon: MessageSquare },
  { key: 'sql', label: 'SQL Queries', icon: Database },
  { key: 'research', label: 'Research History', icon: Globe },
  { key: 'saved', label: 'Saved Reports', icon: Bookmark },
]

export default function Sidebar() {
  const navigate = useNavigate()
  const location = useLocation()
  // Sidebar sits outside the matched <Route> element, so useParams()
  // won't see the :sessionId param — parse it from the path instead.
  const activeSessionId = location.pathname.match(/^\/chat\/([^/]+)/)?.[1]
  const [openSection, setOpenSection] = useState('chat')
  const [sessionsBySection, setSessionsBySection] = useState({})

  useEffect(() => {
    refreshSection(openSection)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openSection])

  async function refreshSection(sectionKey) {
    try {
      const sessions =
        sectionKey === 'saved'
          ? await listSessions({ savedOnly: true })
          : await listSessions({ category: sectionKey })
      setSessionsBySection((prev) => ({ ...prev, [sectionKey]: sessions }))
    } catch (err) {
      console.error('Failed to load sessions:', err)
    }
  }

  async function handleNewChat() {
    const { session_id } = await createSession()
    navigate(`/chat/${session_id}`)
  }

  return (
    <aside className="w-64 h-full bg-card border-r border-border flex flex-col shrink-0">
      <div className="flex items-center gap-2 px-4 py-4 border-b border-border">
        <Sparkles size={18} className="text-primary" />
        <span className="font-semibold text-sm text-foreground">AI Business Research</span>
      </div>

      <div className="px-3 pt-3">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
                     bg-primary text-white hover:bg-primary/90 transition-colors"
        >
          <Plus size={16} />
          New Chat
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-1">
        {NAV_SECTIONS.map((section) => (
          <div key={section.key}>
            <button
              onClick={() => setOpenSection(openSection === section.key ? null : section.key)}
              className="w-full flex items-center gap-2 px-2 py-2 rounded-lg text-sm text-muted
                         hover:bg-background/60 hover:text-foreground transition-colors"
            >
              <section.icon size={15} />
              {section.label}
            </button>

            {openSection === section.key && (
              <div className="ml-6 mt-1 space-y-0.5 border-l border-border pl-2">
                {(sessionsBySection[section.key] || []).length === 0 && (
                  <p className="text-xs text-muted py-1.5">No items yet.</p>
                )}
                {(sessionsBySection[section.key] || []).map((session) => (
                  <button
                    key={session.id}
                    onClick={() => navigate(`/chat/${session.id}`)}
                    className={`w-full text-left truncate text-xs px-2 py-1.5 rounded-md transition-colors ${
                      activeSessionId === session.id
                        ? 'bg-primary/15 text-primary'
                        : 'text-muted hover:bg-background/60 hover:text-foreground'
                    }`}
                    title={session.title}
                  >
                    {session.title}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </nav>

      <div className="px-3 py-3 border-t border-border space-y-1">
        <button className="w-full flex items-center gap-2 px-2 py-2 rounded-lg text-sm text-muted hover:bg-background/60 hover:text-foreground transition-colors">
          <Settings size={15} />
          Settings
        </button>
        <button className="w-full flex items-center gap-2 px-2 py-2 rounded-lg text-sm text-muted hover:bg-background/60 hover:text-foreground transition-colors">
          <LogOut size={15} />
          Logout
        </button>
      </div>
    </aside>
  )
}
