import { Bot, User, Wrench, Zap, Code2, Download } from 'lucide-react'
import SqlResultTable from './SqlResultTable'
import WebResearchCard from './WebResearchCard'
import ChartRenderer from './ChartRenderer'
import { exportTable } from '../lib/api'

const TOOL_BADGE_STYLES = {
  'SQL Agent': 'bg-primary/15 text-primary border-primary/30',
  'Web Research': 'bg-accent/15 text-accent border-accent/30',
  Mixed: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
  'Direct Answer': 'bg-slate-500/15 text-muted border-slate-500/30',
}

export default function ChatMessage({ message }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="flex items-start gap-2 max-w-2xl">
          <div className="bg-primary text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm">
            {message.content}
          </div>
          <div className="w-7 h-7 rounded-full bg-card border border-border flex items-center justify-center shrink-0 mt-0.5">
            <User size={14} className="text-muted" />
          </div>
        </div>
      </div>
    )
  }

  const isSql = message.tool_used === 'SQL Agent' || message.tool_used === 'Mixed'
  const isWeb = message.tool_used === 'Web Research' || message.tool_used === 'Mixed'

  return (
    <div className="flex justify-start mb-4">
      <div className="flex items-start gap-2 max-w-3xl w-full">
        <div className="w-7 h-7 rounded-full bg-primary/20 border border-primary/40 flex items-center justify-center shrink-0 mt-0.5">
          <Bot size={14} className="text-primary" />
        </div>

        <div className="bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-3 text-sm w-full">
          <div className="flex items-center gap-2 mb-2">
            <Bot size={14} className="text-primary" />
            <span className="font-semibold text-foreground">AI Response</span>
          </div>

          <div className="border-t border-border/60 my-2" />

          <p className="text-foreground leading-relaxed whitespace-pre-wrap">{message.content}</p>

          {message.tool_used && (
            <>
              <div className="border-t border-border/60 my-3" />
              <div className="flex flex-wrap items-center gap-3">
                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
                    TOOL_BADGE_STYLES[message.tool_used] || TOOL_BADGE_STYLES['Direct Answer']
                  }`}
                >
                  <Wrench size={11} />
                  {message.tool_used}
                </span>

                {typeof message.execution_time_seconds === 'number' && (
                  <span className="inline-flex items-center gap-1.5 text-xs text-muted">
                    <Zap size={12} className="text-accent" />
                    {message.execution_time_seconds.toFixed(2)} sec
                  </span>
                )}
              </div>
            </>
          )}

          {message.generated_sql && (
            <>
              <div className="border-t border-border/60 my-3" />
              <div className="flex items-center gap-1.5 text-xs font-semibold text-muted uppercase tracking-wide mb-1.5">
                <Code2 size={12} />
                Generated SQL
              </div>
              <pre className="bg-background border border-border rounded-lg px-3 py-2 text-xs text-accent overflow-x-auto">
                {message.generated_sql}
              </pre>
            </>
          )}

          {isSql && message.table_data && <SqlResultTable rows={message.table_data} />}

          {message.chart && <ChartRenderer chart={message.chart} />}

          {isWeb && !isSql && (
            <WebResearchCard answer={message.content} sources={message.sources} />
          )}

          {isSql && message.table_data && (
            <div className="mt-3">
              <button
                onClick={() => exportTable('csv', message.table_data, 'query_results')}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                           bg-primary text-white hover:bg-primary/90 transition-colors"
              >
                <Download size={13} />
                Download CSV
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
