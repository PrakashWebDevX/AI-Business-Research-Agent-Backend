import { ExternalLink, Globe } from 'lucide-react'

/**
 * Renders the "AI Research" card for web-search-backed answers: the
 * answer broken into bullet highlights (best-effort, from the plain
 * text), followed by a numbered list of sources.
 */
export default function WebResearchCard({ answer, sources }) {
  const bullets = extractBullets(answer)

  return (
    <div className="bg-card border border-border rounded-xl p-4 mt-3">
      <div className="flex items-center gap-2 mb-3">
        <Globe size={16} className="text-primary" />
        <span className="text-sm font-semibold text-foreground">AI Research</span>
      </div>

      {bullets.length > 1 ? (
        <ul className="space-y-1.5 mb-3">
          {bullets.map((bullet, i) => (
            <li key={i} className="text-sm text-foreground flex gap-2">
              <span className="text-accent">•</span>
              <span>{bullet}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-foreground mb-3 leading-relaxed">{answer}</p>
      )}

      {sources && sources.length > 0 && (
        <div className="pt-3 border-t border-border">
          <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">Sources</p>
          <ol className="space-y-1">
            {sources.map((source, i) => (
              <li key={i} className="text-xs flex items-start gap-1.5">
                <span className="text-muted">{i + 1}.</span>
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline flex items-center gap-1 truncate"
                >
                  {source.title || source.url}
                  <ExternalLink size={11} className="shrink-0" />
                </a>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}

/**
 * Best-effort split of a plain-text answer into bullet points, so
 * multi-point research answers render as a scannable list rather than
 * a wall of text. Falls back to treating the whole answer as one item
 * if it doesn't look like a list.
 */
function extractBullets(text) {
  if (!text) return []
  const lines = text
    .split(/\n|(?<=\.)\s+(?=[A-Z])/)
    .map((line) => line.replace(/^[-•*\d.)\s]+/, '').trim())
    .filter(Boolean)
  return lines.length > 1 ? lines : [text]
}
