/**
 * Presentational building blocks for the console.
 *
 * These components render fields exactly as returned by the backend. They do
 * not derive, average, or infer any value; a missing field renders as an
 * explicit dash rather than a computed placeholder.
 */

export function Card({ title, subtitle, tag, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      <header className="card-head">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p className="card-sub">{subtitle}</p> : null}
        </div>
        {tag ? <span className={`tag tag-${tag.tone}`}>{tag.text}</span> : null}
      </header>
      <div className="card-body">{children}</div>
    </section>
  )
}

export function Row({ label, value }) {
  return (
    <div className="row">
      <span className="row-label">{label}</span>
      <span className="row-value">{value ?? '—'}</span>
    </div>
  )
}

/** Renders a signed percentage exactly as delivered; never recomputed. */
export function Pct({ value, digits = 1 }) {
  if (value === null || value === undefined) return <span>—</span>
  const sign = value > 0 ? '+' : ''
  return <span>{`${sign}${Number(value).toFixed(digits)}%`}</span>
}

export function Num({ value, digits = 2 }) {
  if (value === null || value === undefined) return <span>—</span>
  return <span>{Number(value).toFixed(digits)}</span>
}

export function Badge({ children, tone = 'neutral' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

const STATUS_TONE = {
  supported: 'good',
  rejected: 'bad',
  inconclusive: 'warn',
  complete: 'good',
  failed: 'bad',
  skipped: 'muted',
  low: 'muted',
  medium: 'warn',
  high: 'bad',
}

export function StatusBadge({ status }) {
  const key = String(status || '').toLowerCase()
  return <Badge tone={STATUS_TONE[key] || 'neutral'}>{key || 'unknown'}</Badge>
}