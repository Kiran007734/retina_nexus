import { Check, Circle } from 'lucide-react';

export const SCREENING_STEPS = ['Upload image', 'Quality check', 'AI analysis', 'Clinical evidence', 'RetinaGuard', 'Final decision'];

export function ScreeningStepper({ active, compact = false }: { active: number; compact?: boolean }) {
  return <div className={`flex ${compact ? 'gap-2 overflow-x-auto pb-1' : 'items-start justify-between'} `} aria-label="Screening progress">
    {SCREENING_STEPS.map((step, index) => {
      const complete = index < active;
      const current = index === active;
      return <div key={step} className={`flex min-w-0 items-center ${!compact ? 'flex-1 last:flex-none' : ''}`}><div className="flex min-w-0 items-center gap-2"><span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-extrabold ${complete ? 'bg-teal-500 text-white' : current ? 'border-2 border-teal-500 bg-white text-teal-700' : 'border border-line bg-white text-slate-400'}`}>{complete ? <Check size={14} /> : current ? <span>{index + 1}</span> : <Circle size={12} />}</span><span className={`whitespace-nowrap text-[11px] font-bold ${current || complete ? 'text-ink' : 'text-slate-400'}`}>{step}</span></div>{index < SCREENING_STEPS.length - 1 && <div className={`mx-3 hidden h-px flex-1 ${index < active ? 'bg-teal-300' : 'bg-line'} ${compact ? '!hidden' : 'sm:block'}`} />}</div>;
    })}
  </div>;
}
