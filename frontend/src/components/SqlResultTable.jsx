import { Download, FileSpreadsheet, FileJson } from 'lucide-react'
import { exportTable } from '../lib/api'

/**
 * Renders SQL query result rows as a professional data table, with
 * Export CSV / Export Excel / Download JSON buttons.
 */
export default function SqlResultTable({ rows }) {
  if (!rows || rows.length === 0) return null

  const columns = Object.keys(rows[0])

  const handleExport = (format) => {
    exportTable(format, rows, 'query_results').catch((err) => {
      console.error('Export failed:', err)
    })
  }

  return (
    <div className="bg-card border border-border rounded-xl overflow-hidden mt-3">
      <div className="overflow-x-auto max-h-96">
        <table className="w-full text-sm text-left">
          <thead className="bg-background/60 sticky top-0">
            <tr>
              {columns.map((col) => (
                <th
                  key={col}
                  className="px-4 py-2.5 font-semibold text-foreground border-b border-border whitespace-nowrap"
                >
                  {formatColumnName(col)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={i}
                className="border-b border-border/60 last:border-0 hover:bg-background/40 transition-colors"
              >
                {columns.map((col) => (
                  <td key={col} className="px-4 py-2 text-muted whitespace-nowrap">
                    {formatCell(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center gap-2 px-4 py-3 border-t border-border bg-background/40">
        <ExportButton icon={Download} label="Export CSV" onClick={() => handleExport('csv')} />
        <ExportButton icon={FileSpreadsheet} label="Export Excel" onClick={() => handleExport('excel')} />
        <ExportButton icon={FileJson} label="Download JSON" onClick={() => handleExport('json')} />
      </div>
    </div>
  )
}

function ExportButton({ icon: Icon, label, onClick }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                 bg-card border border-border text-foreground
                 hover:border-primary hover:text-primary transition-colors"
    >
      <Icon size={14} />
      {label}
    </button>
  )
}

function formatColumnName(col) {
  return col.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatCell(value) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return value.toLocaleString()
  return String(value)
}
