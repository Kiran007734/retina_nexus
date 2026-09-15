import { CheckCircle2 } from 'lucide-react';
import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { BrandMark } from './BrandMark';

export function AuthLayout({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children: ReactNode }) {
  return (
    <div className="min-h-screen bg-mist lg:grid lg:grid-cols-[46%_54%]">
      <aside className="relative overflow-hidden bg-ink px-6 py-7 text-white sm:px-10 lg:sticky lg:top-0 lg:flex lg:h-screen lg:flex-col lg:justify-between lg:p-10">
        <div className="pointer-events-none absolute -left-24 top-32 h-96 w-96 rounded-full border border-teal-400/10" />
        <div className="pointer-events-none absolute -left-10 top-48 h-64 w-64 rounded-full border border-teal-400/10" />
        <Link to="/" aria-label="Return to RETINA NEXUS home" className="relative inline-flex"><BrandMark dark /></Link>
        <div className="relative mt-12 lg:mt-0">
          <p className="eyebrow !text-teal-300">The care intelligence layer</p>
          <h1 className="section-title mt-4 text-4xl font-extrabold leading-[1.08] sm:text-5xl">See clearly.<br /><span className="text-teal-300">Screen early.</span><br />Save sight.</h1>
          <p className="mt-5 max-w-sm text-sm leading-6 text-slate-300">A self-checking AI workspace designed to help primary care teams make every retinal screening more explainable and actionable.</p>
          <div className="mt-7 hidden space-y-3 text-sm text-slate-300 sm:block">
            <p className="flex items-center gap-2"><CheckCircle2 size={16} className="text-teal-300" /> Quality-first image intake</p>
            <p className="flex items-center gap-2"><CheckCircle2 size={16} className="text-teal-300" /> Evidence-backed results</p>
            <p className="flex items-center gap-2"><CheckCircle2 size={16} className="text-teal-300" /> Human review when it matters</p>
          </div>
        </div>
        <p className="relative mt-10 hidden text-[11px] text-slate-500 lg:block">RETINA-NEXUS · Prototype workspace</p>
      </aside>

      <main className="flex min-h-[620px] items-center justify-center px-5 py-10 sm:px-12 lg:min-h-screen lg:py-14">
        <div className="w-full max-w-[430px]">
          <p className="eyebrow">{eyebrow}</p>
          <h2 className="section-title mt-3 text-3xl font-extrabold text-ink">{title}</h2>
          <p className="mt-3 text-sm leading-6 text-slate-500">{description}</p>
          {children}
        </div>
      </main>
    </div>
  );
}

export const authInputClass = 'w-full rounded-xl border border-line bg-white px-4 py-3 text-sm text-ink outline-none transition placeholder:text-slate-300 focus:border-teal-400 focus:ring-4 focus:ring-teal-50';

export function FieldError({ message }: { message?: string }) {
  return message ? <p className="mt-1.5 text-[11px] font-semibold text-rose-600">{message}</p> : null;
}
