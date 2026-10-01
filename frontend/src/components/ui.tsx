/**
 * Presentational building blocks for the console.
 *
 * These components render fields exactly as returned by the backend. They do
 * not derive, average, or infer any value; a missing field renders as an
 * explicit dash rather than a computed placeholder.
 */
import type { ReactNode } from 'react'

/** Colour tones shared by badges and card tags. */
export type Tone = 'neutral' | 'good' | 'bad' | 'warn' | 'muted' | 'info'

export interface CardTag {
  text: string
  tone: Tone
}

interface CardProps {
  title: string
  subtitle?: string
  tag?: CardTag
  children: ReactNode
  className?: string
}

export function Card({ title, subtitle, tag, children, className = '' }: CardProps) {
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

interface RowProps {
  label: string
  value: ReactNode
}

export function Row({ label, value }: RowProps) {
  return (
    <div className="row">
      <span className="row-label">{label}</span>
      <span className="row-value">{value ?? '—'}</span>
    </div>
  )
}

interface PctProps {
  value: number | null | undefined
  digits?: number
}

/** Renders a signed percentage exactly as delivered; never recomputed. */
export function Pct({ value, digits = 1 }: PctProps) {
  if (value === null || value === undefined) return <span>—</span>
  const sign = value > 0 ? '+' : ''
  return <span>{`${sign}${Number(value).toFixed(digits)}%`}</span>
}

interface NumProps {
  value: number | null | undefined
  digits?: number
}

export function Num({ value, digits = 2 }: NumProps) {
  if (value === null || value === undefined) return <span>—</span>
  return <span>{Number(value).toFixed(digits)}</span>
}

interface BadgeProps {
  children: ReactNode
  tone?: Tone
}

export function Badge({ children, tone = 'neutral' }: BadgeProps) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

const STATUS_TONE: Record<string, Tone> = {
  supported: 'good',
  rejected: 'bad',
  inconclusive: 'warn',
  complete: 'good',
  no_anomaly: 'good',
  failed: 'bad',
  verification_failed: 'bad',
  skipped: 'muted',
  low: 'muted',
  medium: 'warn',
  high: 'bad',
}

interface StatusBadgeProps {
  status: string | null | undefined
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const key = String(status || '').toLowerCase()
  return <Badge tone={STATUS_TONE[key] || 'neutral'}>{key || 'unknown'}</Badge>
}
