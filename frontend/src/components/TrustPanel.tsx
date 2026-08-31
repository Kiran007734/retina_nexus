import { AlertTriangle, CheckCircle2, CircleAlert, ShieldCheck, XCircle } from 'lucide-react';
import type { TrustResult } from '../services/api';
import { StatusBadge } from './StatusBadge';

const FACTORS = [
  ['quality', 'Image quality'],
  ['calibrated_confidence', 'Confidence'],
  ['attention_lesion_agreement', 'Evidence agreement'],
  ['model_agreement', 'Model agreement'],
  ['explanation_stability', 'Explanation stability'],
  ['ood', 'Distribution check'],
] as const;

type State = 'TRUSTED' | 'REVIEW_RECOMMENDED' | 'UNRELIABLE' | 'INSUFFICIENT_EVIDENCE' | 'UNCERTAIN';

function stateMeta(category: string): { title: string; tone: 'success' | 'warning' | 'danger'; Icon: typeof CheckCircle2; colour: string } {
  if (category === 'TRUSTED') return { title: 'Trusted for AI triage', tone: 'success', Icon: CheckCircle2, colour: 'text-emerald-600' };
  if (category === 'UNRELIABLE') return { title: 'Automated interpretation is unreliable', tone: 'danger', Icon: XCircle, colour: 'text-rose-600' };
  if (category === 'INSUFFICIENT_EVIDENCE') return { title: 'Required evidence is unavailable', tone: 'warning', Icon: CircleAlert, colour: 'text-amber-600' };
  return { title: 'Professional review recommended', tone: 'warning', Icon: AlertTriangle, colour: 'text-amber-600' };
}

export function TrustPanel({ result }: { result: TrustResult | null }) {
  if (!result) return <div className="surface border-amber-200 bg-amber-50/60 p-6"><p className="eyebrow text-amber-700">RetinaGuard</p><h3 className="section-title mt-2 text-lg font-extrabold text-amber-950">Not available for this run</h3><p className="mt-2 text-sm leading-6 text-amber-900">The self-check did not produce a reliability state. Review the run status before relying on any AI output.</p></div>;
  const meta = stateMeta(result.trust_category as State);
  const factorMap = new Map(result.contributing_factors.map((factor) => [factor.factor, factor]));
  const cardTone = meta.tone === 'success' ? 'border-emerald-200 bg-emerald-50/45' : meta.tone === 'danger' ? 'border-rose-200 bg-rose-50/45' : 'border-amber-200 bg-amber-50/55';
  return <div className={`overflow-hidden rounded-2xl border shadow-soft ${cardTone}`}>
    <div className="flex flex-col justify-between gap-4 border-b border-black/5 px-6 py-5 sm:flex-row sm:items-start">
      <div><div className="flex items-center gap-2"><meta.Icon size={20} className={meta.colour} /><p className="eyebrow !text-slate-500">RetinaGuard self-check</p></div><h3 className="section-title mt-2 text-2xl font-extrabold text-ink">{meta.title}</h3><p className="mt-1 text-xs leading-5 text-slate-600">Engineering operating decision - not a clinical trust guarantee</p></div>
      <StatusBadge tone={meta.tone}>{result.trust_category}</StatusBadge>
    </div>
    <div className="px-6 py-6">
      <div className="flex items-end justify-between gap-4"><div><p className="text-5xl font-extrabold tracking-tight text-ink">{Math.round(result.trust_score * 100)}<span className="text-lg text-slate-400">/100</span></p><p className="mt-1 text-xs font-semibold text-slate-500">Transparent composite score</p></div><ShieldCheck size={34} className={meta.colour} /></div>
      <div className="mt-5"><div className="relative h-2 rounded-full bg-gradient-to-r from-rose-300 via-amber-300 to-emerald-400"><span className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-ink shadow" style={{ left: `${Math.max(0, Math.min(100, result.trust_score * 100))}%` }} /></div><div className="mt-2 flex justify-between text-[10px] font-bold uppercase tracking-widest text-slate-500"><span>Low</span><span>Medium</span><span>High</span></div></div>
      <div className="mt-6 grid gap-2 sm:grid-cols-2">{FACTORS.map(([key, label]) => { const factor = factorMap.get(key); const flagged = result.risk_flags.some((flag) => (key === 'model_agreement' && flag.code.includes('disagreement')) || (key === 'attention_lesion_agreement' && flag.code.includes('agreement')) || (key === 'ood' && flag.code.includes('shift')) || (key === 'explanation_stability' && flag.code.includes('stability'))); return <div key={key} className="flex items-center gap-3 rounded-xl border border-black/5 bg-white/65 p-3"><span className={flagged ? 'text-amber-600' : factor?.status === 'available' ? 'text-emerald-600' : 'text-slate-400'}>{flagged ? <CircleAlert size={16} /> : factor?.status === 'available' ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}</span><span className="min-w-0 flex-1 text-xs font-bold text-ink">{label}</span><span className="text-xs font-extrabold text-ink">{factor?.score == null ? '-' : `${Math.round(factor.score * 100)}%`}</span></div>; })}</div>
      <div className="mt-5 rounded-xl bg-white/70 p-4"><p className="text-xs font-extrabold text-ink">Safe action</p><p className="mt-1 text-sm font-semibold leading-6 text-slate-600">{result.recommended_safe_action ?? result.recommended_action}</p><p className="mt-1 text-xs leading-5 text-slate-500">{result.recommended_action}</p></div>
      {(result.reason_summary.length > 0 || (result.reasons?.length ?? 0) > 0) && <div className="mt-5 border-t border-black/5 pt-4"><p className="eyebrow">Why this decision</p><ul className="mt-2 space-y-1.5 text-xs leading-5 text-slate-600">{(result.reasons ?? result.reason_summary).map((reason) => <li key={reason} className="flex gap-2"><span className="text-slate-400">-</span>{reason}</li>)}</ul></div>}
      <p className="mt-5 border-t border-black/5 pt-4 text-[11px] leading-5 text-slate-500">Signals: quality {result.image_quality_status ?? 'UNAVAILABLE'}, evidence {result.evidence_status ?? 'UNAVAILABLE'}, explanation {result.explanation_status ?? 'UNAVAILABLE'}, OOD {result.ood_status ?? 'UNAVAILABLE'}.</p>
    </div>
  </div>;
}
