import { Activity, ArrowUpRight, ShieldCheck, Sparkles } from 'lucide-react';
import { StatusBadge } from './StatusBadge';

export function AIStatusCard() {
  return (
    <div className="relative overflow-hidden rounded-2xl bg-ink p-5 text-white shadow-soft">
      <div className="absolute -right-14 -top-16 h-40 w-40 rounded-full border border-teal-400/20" />
      <div className="absolute -right-5 -top-7 h-24 w-24 rounded-full border border-teal-400/20" />
      <div className="relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2"><div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-400/15 text-teal-300"><Sparkles size={16} /></div><span className="text-sm font-bold">AI system status</span></div>
          <StatusBadge tone="warning" dot>Model setup pending</StatusBadge>
        </div>
        <p className="mt-5 max-w-[230px] text-[13px] leading-5 text-slate-300">The interface is ready for image intake. No clinical model is active in this local scaffold.</p>
        <div className="mt-5 grid grid-cols-2 gap-2 border-t border-white/10 pt-4">
          <div><p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Model</p><p className="mt-1 text-sm font-bold">Not configured</p></div>
          <div><p className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Latency</p><p className="mt-1 text-sm font-bold">—</p></div>
        </div>
        <div className="mt-4 flex items-center gap-3 text-xs text-slate-400"><ShieldCheck size={14} className="text-teal-300" /> Quality + evidence interfaces staged <ArrowUpRight size={14} className="ml-auto text-teal-300" /></div>
      </div>
    </div>
  );
}
