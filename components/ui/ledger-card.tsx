import type { HTMLAttributes, ReactNode } from 'react'

type LedgerCardProps = HTMLAttributes<HTMLDivElement> & {
  children: ReactNode
  as?: 'div' | 'section' | 'article'
}

export function LedgerCard({ children, className = '', as = 'div', ...props }: LedgerCardProps) {
  const Card = as
  return <Card {...props} className={`ledger-card ${className}`}>{children}</Card>
}
