import { Eye, Info, LoaderCircle, SlidersHorizontal } from 'lucide-react';
import { useState } from 'react';
import { imageContentUrl, type EvidenceAnalysisResult } from '../services/api';
import { StatusBadge } from './StatusBadge';

type EvidenceViewerProps = {
  imageId: string;
  evidence: EvidenceAnalysisResult | null;
  loading: boolean;
  error: string;
};

export function EvidenceViewer({ imageId, evidence, loading, error }: EvidenceViewerProps) {
  const [visible, setVisible] = useState({ vessels: true, lesions: true, opticDisc: true, other: false });
  const [opacity, setOpacity] = useState(0.72);
  const vessel = evidence?.modules.vessel_segmentation;
  const opticDisc = evidence?.modules.optic_disc_localization;
  const lesionModules = evidence ? ['cotton_wool_spot_detection', 'microaneurysm_detection', 'hemorrhage_detection', 'exudate_segmentation', 'neovascularization_detection'].map((key) => evidence.modules[key]).filter(Boolean) : [];
  const hasModelEvidence = [vessel, ...lesionModules].some((module) => module?.status === 'model_inference');
  const fovea = evidence?.anatomical_landmarks.find((landmark) => landmark.landmark_type === 'fovea');

  return <div className="card overflow-hidden"><div className="flex flex-col justify-between gap-3 border-b border-line px-5 py-4 sm:flex-row sm:items-center"><div><p className="eyebrow">Clinical evidence</p><h3 className="section-title mt-1 text-base font-extrabold">Evidence viewer</h3></div>{loading ? <StatusBadge tone="teal"><LoaderCircle size={13} className="animate-spin" /> Analyzing structures</StatusBadge> : evidence ? <StatusBadge tone={hasModelEvidence ? 'success' : 'neutral'}>{hasModelEvidence ? 'Model-backed evidence' : 'Experimental / unavailable'}</StatusBadge> : <StatusBadge tone="neutral">Not available</StatusBadge>}</div>
    {error && <div className="border-b border-amber-100 bg-amber-50 px-5 py-3 text-xs leading-5 text-amber-800">{error}</div>}
    <div className="p-5">
      <div className="relative overflow-hidden rounded-2xl bg-[#132643]"><img src={imageContentUrl(imageId)} alt="Original fundus image" className="block max-h-[470px] min-h-[260px] w-full object-contain" />
        {visible.vessels && vessel?.mask_data_uri && <img src={vessel.mask_data_uri} alt="Vessel overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity }} />}
        {visible.lesions && lesionModules.map((module) => module.mask_data_uri && <img key={module.module} src={module.mask_data_uri} alt={`${module.module} overlay`} className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity }} />)}
        {visible.opticDisc && opticDisc?.mask_data_uri && <img src={opticDisc.mask_data_uri} alt="Optic disc overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity }} />}
        {visible.other && evidence?.evidence_map_data_uri && <img src={evidence.evidence_map_data_uri} alt="Combined evidence overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity: opacity * 0.35 }} />}
        {visible.other && fovea && <span className="pointer-events-none absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-cyan-300" style={{ left: `${Number(fovea.x_normalized) * 100}%`, top: `${Number(fovea.y_normalized) * 100}%` }} />}
        <span className="absolute left-3 top-3 rounded-lg bg-black/35 px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-widest text-white/80">Original image</span>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-2"><Toggle active={visible.vessels} onClick={() => setVisible((current) => ({ ...current, vessels: !current.vessels }))} colour="bg-emerald-400">Vessels</Toggle><Toggle active={visible.lesions} onClick={() => setVisible((current) => ({ ...current, lesions: !current.lesions }))} colour="bg-rose-400">Lesions</Toggle><Toggle active={visible.opticDisc} onClick={() => setVisible((current) => ({ ...current, opticDisc: !current.opticDisc }))} colour="bg-blue-400">Optic disc</Toggle><Toggle active={visible.other} onClick={() => setVisible((current) => ({ ...current, other: !current.other }))} colour="bg-amber-300">Other evidence</Toggle></div>
      <label className="mt-4 flex items-center gap-3 text-xs text-slate-500"><SlidersHorizontal size={14} className="text-teal-600" /><span className="font-semibold">Overlay opacity</span><input aria-label="Overlay opacity" type="range" min="0" max="1" step="0.05" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} className="min-w-32 flex-1 accent-teal-600" /><span className="w-10 text-right font-bold">{Math.round(opacity * 100)}%</span></label>
      <div className="mt-4 flex items-start gap-2 border-t border-line pt-4 text-[11px] leading-5 text-slate-400"><Info size={14} className="mt-0.5 shrink-0 text-teal-600" /><span>{evidence?.note ?? 'Structure analysis will appear here after the Trust Gate.'}</span></div>
      {evidence && <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">{vessel && <div className="rounded-xl bg-emerald-50 p-3"><p className="truncate text-[10px] font-bold uppercase tracking-wider text-emerald-700">Vessel segmentation</p><p className="mt-1 text-sm font-extrabold text-ink">{vessel.status === 'model_inference' ? 'Real model output' : vessel.status.replaceAll('_', ' ')}</p><p className="mt-1 text-[10px] text-emerald-800">{typeof vessel.metadata.coverage_ratio === 'number' ? `${(vessel.metadata.coverage_ratio * 100).toFixed(1)}% engineering coverage estimate` : vessel.implementation}</p></div>}{lesionModules.filter((module) => module.category === 'lesion_detection').map((module) => <div key={module.module} className="rounded-xl bg-mist p-3"><p className="truncate text-[10px] font-bold uppercase tracking-wider text-slate-400">{module.module.replaceAll('_', ' ')}</p><p className="mt-1 text-sm font-extrabold text-ink">{module.count ?? '—'}</p><p className="mt-1 text-[10px] text-slate-400">{module.status.replaceAll('_', ' ')}</p></div>)}</div>}
    </div>
  </div>;
}

function Toggle({ active, onClick, colour, children }: { active: boolean; onClick: () => void; colour: string; children: React.ReactNode }) {
  return <button onClick={onClick} className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-2 text-[11px] font-bold transition ${active ? 'border-line bg-white text-ink' : 'border-transparent bg-mist text-slate-400'}`}><span className={`h-2 w-2 rounded-full ${colour} ${active ? '' : 'opacity-35'}`} />{active ? <Eye size={13} /> : null}{children}</button>;
}
