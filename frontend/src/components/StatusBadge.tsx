import { CheckCircle2, CircleAlert, Clock3, XCircle } from 'lucide-react';
import type { StatusTone } from '../types';

const toneClasses: Record<StatusTone, string> = {
  success: 'bg-emerald-50 text-emerald-700 ring-emerald-100',
  warning: 'bg-amber-50 text-amber-700 ring-amber-100',
  danger: 'bg-rose-50 text-rose-700 ring-rose-100',
  neutral: 'bg-slate-100 text-slate-600 ring-slate-200',
  teal: 'bg-teal-50 text-teal-700 ring-teal-100',
};

const toneIcons: Record<StatusTone, typeof CheckCircle2> = {
  success: CheckCircle2,
  warning: CircleAlert,
  danger: XCircle,
  neutral: Clock3,
  teal: CheckCircle2,
};

export function StatusBadge({ children, tone = 'neutral', dot = false }: { children: React.ReactNode; tone?: StatusTone; dot?: boolean }) {
  const Icon = toneIcons[tone];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-bold ring-1 ring-inset ${toneClasses[tone]}`}>
      {dot ? <span className="h-1.5 w-1.5 rounded-full bg-current" /> : <Icon size={13} />}
      {children}
    </span>
  );
}
