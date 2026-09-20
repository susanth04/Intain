"use client"

import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { motion } from "framer-motion"
import { ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"

const cardVariants = cva(
  "relative flex min-h-[164px] flex-col justify-between overflow-hidden rounded-2xl border p-5 shadow-sm transition-shadow duration-300 ease-out group hover:shadow-xl",
  {
    variants: {
      variant: {
        default: "border-cyan-200/20 bg-cyan-400/10 text-slate-50",
        red: "border-orange-300/30 bg-orange-400/15 text-orange-50",
        blue: "border-sky-300/25 bg-sky-400/15 text-sky-50",
        gray: "border-slate-300/20 bg-slate-500/15 text-slate-50",
      },
    },
    defaultVariants: { variant: "default" },
  },
)

export interface ServiceCardProps
  extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof cardVariants> {
  title: string
  href: string
  imgSrc?: string
  imgAlt?: string
  value?: React.ReactNode
  description?: string
}

const ServiceCard = React.forwardRef<HTMLDivElement, ServiceCardProps>(
  ({ className, variant, title, href, imgSrc, imgAlt, value, description, ...props }, ref) => {
    return (
      <motion.div
        ref={ref}
        className={cn(cardVariants({ variant, className }))}
        whileHover={{ scale: 1.018 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
        {...props}
      >
        <div className="relative z-10 flex h-full flex-col">
          <span className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] opacity-65">Portfolio signal</span>
          <h3 className="text-xl font-semibold tracking-tight">{title}</h3>
          {value && <div className="mt-2 text-4xl font-semibold tracking-tight">{value}</div>}
          {description && <p className="mt-2 max-w-[18rem] text-xs leading-5 opacity-70">{description}</p>}
          <a href={href} aria-label={`Open ${title}`} className="mt-auto flex items-center pt-4 text-[10px] font-bold uppercase tracking-[0.16em] opacity-75 transition-opacity group-hover:opacity-100">
            Inspect signal
            <motion.span className="inline-flex" whileHover={{ x: 5 }}>
              <ArrowRight className="ml-2 h-4 w-4" />
            </motion.span>
          </a>
        </div>
        {imgSrc && <motion.img src={imgSrc} alt={imgAlt || ""} className="pointer-events-none absolute -bottom-10 -right-8 h-40 w-40 object-contain opacity-65" whileHover={{ scale: 1.1, rotate: 3, x: 10 }} transition={{ duration: 0.4, ease: "easeInOut" }} />}
      </motion.div>
    )
  },
)

ServiceCard.displayName = "ServiceCard"

export { ServiceCard }
