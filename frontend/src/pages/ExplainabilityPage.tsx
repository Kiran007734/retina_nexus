import { ArrowLeft, CircleAlert, Info, LoaderCircle, SlidersHorizontal, Sparkles } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { StatusBadge } from '../components/StatusBadge';
import { explainImage, imageContentUrl, type ExplainabilityResult, type ScreeningRun } from '../services/api';

type NavigationState = { imageId?: string; screeningSessionId?: string; run?: ScreeningRun };

export function ExplainabilityPage() {
  const location = useLocation();
  const navigation = (location.state as NavigationState | null) ?? null;
  const imageId = navigation?.imageId ?? navigation?.run?.image_id;
  const [explanation, setExplanation] = useState<ExplainabilityResult | null>(navigation?.run?.explainability ?? null);
  const [loading, setLoading] = useState(Boolean(imageId));
  const [error, setError] = useState('');
  const [runStability, setRunStability] = useState(false);
  const [runCounterfactual, setRunCounterfactual] = useState(false);
  const [visible, setVisible] = useState({ gradCam: true, lesions: true, other: false });
  const [opacity, setOpacity] = useState(0.78);

  useEffect(() => {
    let active = true;
    if (!imageId || (navigation?.run?.explainability && !runStability && !runCounterfactual)) {
      setLoading(false);
      return () => { active = false; };
    }
    setLoading(true);
    setError('');
    explainImage(imageId, navigation?.screeningSessionId, runStability, runCounterfactual)
      .then((result) => { if (active) setExplanation(result); })
      .catch((requestError) => { if (active) setError(requestError instanceof Error ? requestError.message : 'Explainability is unavailable.'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [imageId, navigation?.screeningSessionId, runStability, runCounterfactual]);

  const agreementScore = explanation?.attention_lesion_agreement.score;
  const stability = explanation?.explanation_stability;
  const counterfactual = explanation?.counterfactual;

  return <div className="space-y-6">
    <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
      <div className="flex items-center gap-3"><Link to="/screening/results" state={{ imageId }} aria-label="Back to screening result" className="rounded-xl border border-line bg-white p-2.5 text-slate-500 hover:border-teal-300 hover:text-teal-600"><ArrowLeft size={17} /></Link><div><p className="text-xs font-semibold text-slate-400">Screening {'->'} explanation and verification</p><h2 className="section-title mt-1 text-xl font-extrabold">Explainability</h2></div></div>
      <div className="flex flex-wrap gap-2"><label className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-line bg-white px-3 py-2 text-xs font-bold text-slate-600"><input type="checkbox" checked={runStability} onChange={(event) => setRunStability(event.target.checked)} className="accent-teal-600" /> Run stability test</label><label className="inline-flex cursor-pointer items-center gap-2 rounded-xl border border-line bg-white px-3 py-2 text-xs font-bold text-slate-600"><input type="checkbox" checked={runCounterfactual} onChange={(event) => setRunCounterfactual(event.target.checked)} className="accent-teal-600" /> Run counterfactual</label></div>
    </div>
    <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-800"><div className="flex items-start gap-2"><CircleAlert size={16} className="mt-0.5 shrink-0" /><p><strong>Explainability boundary:</strong> Grad-CAM, attention agreement, stability, and counterfactual outputs are engineering diagnostics. They do not prove clinical causality or provide a clinical trust guarantee.</p></div></div>
    {!imageId && <div className="card p-8 text-center"><p className="eyebrow">No image selected</p><p className="mt-2 text-sm text-slate-500">Open Explainability from a completed screening result.</p></div>}
    {error && <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs leading-5 text-rose-800"><strong>Explainability unavailable.</strong><p className="mt-1">{error}</p><p className="mt-1">Configure a compatible classifier artifact before generating model-linked Grad-CAM.</p></div>}
    {imageId && <div className="grid gap-5 xl:grid-cols-[1fr_390px]">
      <div className="space-y-5">
        <div className="card overflow-hidden"><div className="flex items-center justify-between border-b border-line px-5 py-4"><div><p className="eyebrow">Attention map</p><h3 className="section-title mt-1 text-base font-extrabold">Evidence-linked visual explanation</h3></div>{loading ? <StatusBadge tone="teal"><LoaderCircle size={13} className="animate-spin" /> Generating</StatusBadge> : explanation ? <StatusBadge tone="success">Model-linked Grad-CAM</StatusBadge> : <StatusBadge tone="neutral">Not available</StatusBadge>}</div>
          <div className="p-5"><div className="relative overflow-hidden rounded-2xl bg-[#132643]"><img src={imageContentUrl(imageId)} alt="Original fundus image" className="block max-h-[520px] min-h-[300px] w-full object-contain" />{visible.gradCam && explanation?.grad_cam.heatmap_data_uri && <img src={explanation.grad_cam.heatmap_data_uri} alt="Grad-CAM heatmap overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity }} />}{visible.lesions && explanation?.lesion_evidence_map_data_uri && <img src={explanation.lesion_evidence_map_data_uri} alt="Lesion evidence overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity: opacity * 0.8 }} />}{visible.other && counterfactual?.masked_region_data_uri && <img src={counterfactual.masked_region_data_uri} alt="Counterfactual masked region" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity: 0.7 }} />}<span className="absolute left-3 top-3 rounded-lg bg-black/35 px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-widest text-white/80">Original image</span></div>
            <div className="mt-4 flex flex-wrap items-center gap-2"><Toggle active={visible.gradCam} onClick={() => setVisible((current) => ({ ...current, gradCam: !current.gradCam }))} colour="bg-rose-400">Grad-CAM overlay</Toggle><Toggle active={visible.lesions} onClick={() => setVisible((current) => ({ ...current, lesions: !current.lesions }))} colour="bg-amber-300">Lesion evidence</Toggle><Toggle active={visible.other} onClick={() => setVisible((current) => ({ ...current, other: !current.other }))} colour="bg-cyan-300">Other evidence</Toggle></div>
            <label className="mt-4 flex items-center gap-3 text-xs text-slate-500"><SlidersHorizontal size={14} className="text-teal-600" /><span className="font-semibold">Overlay opacity</span><input aria-label="Explainability overlay opacity" type="range" min="0" max="1" step="0.05" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} className="min-w-32 flex-1 accent-teal-600" /><span className="w-10 text-right font-bold">{Math.round(opacity * 100)}%</span></label>
            {explanation && <p className="mt-4 flex items-start gap-2 border-t border-line pt-4 text-[11px] leading-5 text-slate-400"><Info size={14} className="mt-0.5 shrink-0 text-teal-600" /> Target class: <strong className="text-slate-500">{explanation.predicted_class_label}</strong>. Spatial map layer: {explanation.grad_cam.target_layer}.</p>}
          </div>
        </div>
        <div className="card p-5"><div className="flex items-center gap-2"><Sparkles size={17} className="text-teal-600" /><div><p className="eyebrow">Evidence verification</p><h3 className="section-title mt-1 text-base font-extrabold">Attention ↔ lesion agreement</h3></div></div><div className="mt-4 flex flex-col justify-between gap-4 sm:flex-row sm:items-end"><div><p className="text-3xl font-extrabold text-ink">{agreementScore == null ? '—' : `${Math.round(agreementScore * 100)}%`}</p><p className="mt-1 text-xs text-slate-500">Engineering overlap score</p></div><StatusBadge tone={agreementTone(explanation?.attention_lesion_agreement.status)}>{explanation?.attention_lesion_agreement.status ?? 'Pending'}</StatusBadge></div>{explanation && <div className="mt-5 grid gap-2 sm:grid-cols-3">{[['IoU', explanation.attention_lesion_agreement.metrics.intersection_over_union], ['Dice', explanation.attention_lesion_agreement.metrics.dice], ['Attention in lesion', explanation.attention_lesion_agreement.metrics.attention_in_lesion]].map(([label, value]) => <div key={String(label)} className="rounded-xl bg-mist p-3"><p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{label}</p><p className="mt-1 text-sm font-extrabold text-ink">{value == null ? 'Unavailable' : `${Math.round(Number(value) * 100)}%`}</p></div>)}</div>}<p className="mt-4 text-[11px] leading-5 text-slate-400">{explanation?.attention_lesion_agreement.reason ?? 'Agreement will be calculated from the classifier attention and retinal evidence regions.'}</p></div>
      </div>
      <div className="space-y-5">
        <div className="card p-5"><p className="eyebrow">Classification target</p><h3 className="section-title mt-1 text-2xl font-extrabold">{explanation?.predicted_class_label ?? (loading ? 'Analyzing' : 'Unavailable')}</h3><div className="mt-4 grid grid-cols-2 gap-3"><Metric label="Level" value={explanation ? String(explanation.predicted_class) : '—'} /><Metric label="Model" value={explanation?.model_version ?? '—'} /></div><p className="mt-4 text-xs leading-5 text-slate-500">Grad-CAM is generated for the predicted class from the registered DR classifier artifact.</p></div>
        <div className="card p-5"><p className="eyebrow">Explanation stability</p><h3 className="section-title mt-1 text-base font-extrabold">Controlled perturbations</h3><div className="mt-4 space-y-3"><Summary label="Status" value={stability?.status ?? 'Pending'} /><Summary label="Prediction stability" value={stability?.prediction_stability == null ? 'Not run' : `${Math.round(stability.prediction_stability * 100)}%`} /><Summary label="Grad-CAM stability" value={stability?.grad_cam_stability == null ? 'Not run' : `${Math.round(stability.grad_cam_stability * 100)}%`} /></div><p className="mt-4 text-[11px] leading-5 text-slate-400">{stability?.reason ?? stability?.note ?? 'Disabled by default to keep real-time screening lightweight.'}</p></div>
        <div className="card p-5"><p className="eyebrow">Counterfactual</p><h3 className="section-title mt-1 text-base font-extrabold">Suspicious-region masking</h3><div className="mt-4 space-y-3"><Summary label="Status" value={counterfactual?.status ?? 'Pending'} /><Summary label="Selected region" value={counterfactual?.selected_region?.replaceAll('_', ' ') ?? 'Not run'} /><Summary label="Grade changed" value={counterfactual?.predicted_grade_changed == null ? 'Not run' : counterfactual.predicted_grade_changed ? 'Yes' : 'No'} /></div><p className="mt-4 text-[11px] leading-5 text-slate-400">{counterfactual?.reason ?? counterfactual?.note ?? 'Experimental and disabled by default.'}</p></div>
      </div>
    </div>}
  </div>;
}

function Toggle({ active, onClick, colour, children }: { active: boolean; onClick: () => void; colour: string; children: React.ReactNode }) {
  return <button onClick={onClick} className={`inline-flex items-center gap-2 rounded-lg border px-2.5 py-2 text-[11px] font-bold transition ${active ? 'border-line bg-white text-ink' : 'border-transparent bg-mist text-slate-400'}`}><span className={`h-2 w-2 rounded-full ${colour} ${active ? '' : 'opacity-35'}`} />{children}</button>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl bg-mist p-3"><p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{label}</p><p className="mt-1 truncate text-sm font-extrabold text-ink">{value}</p></div>;
}

function Summary({ label, value }: { label: string; value: string }) {
  return <div className="flex items-center justify-between gap-3 border-b border-line pb-3 last:border-0 last:pb-0"><span className="text-xs text-slate-500">{label}</span><span className="text-right text-xs font-bold capitalize text-ink">{value}</span></div>;
}

function agreementTone(status?: string): 'success' | 'warning' | 'danger' | 'neutral' {
  if (status === 'HIGH AGREEMENT') return 'success';
  if (status === 'MODERATE AGREEMENT') return 'warning';
  if (status === 'LOW AGREEMENT') return 'danger';
  return 'neutral';
}
