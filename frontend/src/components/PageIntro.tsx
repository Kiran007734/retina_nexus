import type { ReactNode } from 'react';

export function PageIntro({ eyebrow, title, description, action }: { eyebrow?: string; title: string; description?: string; action?: ReactNode }) {
  return <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end"><div>{eyebrow && <p className="eyebrow text-teal-700">{eyebrow}</p>}<h2 className="section-title mt-1 text-2xl font-extrabold tracking-tight text-ink sm:text-3xl">{title}</h2>{description && <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">{description}</p>}</div>{action && <div className="flex shrink-0 flex-wrap gap-2">{action}</div>}</div>;
}
