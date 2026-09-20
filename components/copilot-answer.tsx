'use client'

function inline(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, index) =>
    part.startsWith('**') && part.endsWith('**') ? <strong key={index}>{part.slice(2, -2)}</strong> : <span key={index}>{part}</span>,
  )
}

export default function CopilotAnswer({ text }: { text: string }) {
  const lines = text.split('\n')
  return (
    <div className="copilot-prose">
      {lines.map((line, index) => {
        if (line.startsWith('## ')) return <h3 key={index}>{line.slice(3)}</h3>
        if (line.startsWith('### ')) return <h4 key={index}>{line.slice(4)}</h4>
        if (line.startsWith('- ') || line.startsWith('* ')) return <li key={index}>{inline(line.slice(2))}</li>
        if (!line.trim()) return <div key={index} className="copilot-gap" />
        return <p key={index}>{inline(line)}</p>
      })}
    </div>
  )
}
