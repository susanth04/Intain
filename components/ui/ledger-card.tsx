import type { ReactNode } from 'react'

type LedgerCardProps = {
  children: ReactNode
  className?: string
  as?: 'div' | 'section' | 'article'
}

export function LedgerCard({ children, className = '', as = 'div' }: LedgerCardProps) {
  const Card = as
  return <Card className={`ledger-card ${className}`}>{children}</Card>
}
