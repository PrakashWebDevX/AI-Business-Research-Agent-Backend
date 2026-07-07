import { UserCircle } from 'lucide-react'

export default function TopBar() {
  return (
    <header className="h-14 border-b border-border bg-card/60 backdrop-blur flex items-center justify-between px-6 shrink-0">
      <h1 className="text-sm font-semibold text-foreground">AI Business Research Agent</h1>

      <button className="flex items-center gap-2 text-sm text-muted hover:text-foreground transition-colors">
        <UserCircle size={20} />
        <span>User Profile</span>
      </button>
    </header>
  )
}
