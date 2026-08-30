import type { LucideIcon } from 'lucide-react';
import { ArrowDownRight, ArrowUpRight } from 'lucide-react';

export function MetricCard({ label, value, detail, trend, icon: Icon, iconClass = 'bg-teal-50 text-teal-600' }: { label: string; value: string; detail: string; trend?: string; icon: LucideIcon; iconClass?: string }) {
  const down = trend?.startsWith('-');
  return (
    <div className="card group p-5 transition hover:-translate-y-0.5 hover:shadow-lift">
      <div className="flex items-start justify-between">
        <div>
          <p className="eyebrow">{label}</p>
          <p className="section-title mt-3 text-[27px] font-extrabold text-ink">{value}</p>
        </div>
        <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${iconClass}`}><Icon size={19} /></div>
      </div>
      <div className="mt-4 flex items-center gap-2 text-xs">
        {trend && <span className={`inline-flex items-center font-bold ${down ? 'text-rose-500' : 'text-emerald-600'}`}>{down ? <ArrowDownRight size={14} /> : <ArrowUpRight size={14} />}{trend.replace('-', '')}</span>}
        <span className="text-slate-400">{detail}</span>
      </div>
    </div>
  );
}
