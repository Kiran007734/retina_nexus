import { AlertTriangle, ArrowLeft, CheckCircle2, Clock3, Download, FileText, Info, LoaderCircle, Sparkles, XCircle } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { DataState, ErrorState } from '../components/DataState';
import { PageIntro } from '../components/PageIntro';
import { ScreeningStepper } from '../components/ScreeningStepper';
import { StatusBadge } from '../components/StatusBadge';
import { TrustPanel } from '../components/TrustPanel';
import { generateReport, getScreeningRun, imageContentUrl, type ScreeningRun } from '../services/api';

type NavigationState = { imageId?: string; screeningId?: string; run?: ScreeningRun };

export function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const navigation = (location.state as NavigationState | null) ?? null;
  const screeningId = navigation?.screeningId ?? navigation?.run?.screening_id;
  const [imageId, setImageId] = useState(navigation?.imageId ?? navigation?.run?.image_id);
  const [run, setRun] = useState<ScreeningRun | null>(navigation?.run ?? null);
  const [loading, setLoading] = useState(!navigation?.run && Boolean(screeningId));
  const [error, setError] = useState('');
  const [reporting, setReporting] = useState(false);

  useEffect(() => {
    let active = true;
    if (!screeningId || navigation?.run) return () => { active = false; };
    setLoading(true); setError('');
    getScreeningRun(screeningId).then((next) => { if (active) { setRun(next); setImageId(next.image_id); } }).catch((requestError) => { if (active) setError(requestError instanceof Error ? requestError.message : 'Unable to load this screening run.'); }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [screeningId, navigation?.run]);

  useEffect(() => {
    if (!screeningId || !run || run.primary_status !== 'COMPLETED' || run.evidence_status !== 'PROCESSING') return;
    let active = true;
    const refresh = () => getScreeningRun(screeningId).then((next) => { if (active) setRun(next); }).catch(() => { /* Keep the primary result visible if a background poll is interrupted. */ });
    const timer = window.setInterval(refresh, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, [screeningId, run?.primary_status, run?.evidence_status]);

  async function exportReport() {
    if (!screeningId) return;
    setReporting(true);
    try { const report = await generateReport(screeningId); window.open(`${import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1'}${report.download_url ?? `/reports/${report.report_id}/pdf`}`, '_blank', 'noopener,noreferrer'); } catch (requestError) { setError(requestError instanceof Error ? requestError.message : 'Unable to generate the report.'); } finally { setReporting(false); }
  }

  const quality = run?.quality?.final as { quality_decision?: string; quality_score?: number; component_scores?: Record<string, number>; issues?: Array<{ type: string; message: string }> } | undefined;
  const classification = run?.classification;
  const trust = run?.retinaguard;
  const step = run ? run.status === 'COMPLETED' ? 5 : 2 : 0;
  return <div className="space-y-6">
    <div className="flex items-center justify-between gap-3"><Link to="/screening/new" className="btn-quiet !px-0"><ArrowLeft size={15} /> New screening</Link>{run && <StatusBadge tone={run.status === 'FAILED' ? 'danger' : run.status === 'COMPLETED' ? 'success' : 'teal'}>{run.status}</StatusBadge>}</div>
    <PageIntro eyebrow="Screening result" title={classification?.predicted_grade_label ?? (run?.status === 'FAILED' ? 'Run failed safely' : 'Analysis in progress')} description="Review the AI screening recommendation, clinical evidence, and self-check signals together before deciding what happens next." action={<><button onClick={exportReport} disabled={!screeningId || reporting || run?.status !== 'COMPLETED'} className="btn-secondary disabled:opacity-50"><Download size={15} /> {reporting ? 'Preparing PDF…' : 'Export report'}</button>{screeningId && <Link to="/review" state={{ screeningId, imageId }} className="btn-primary"><FileText size={15} /> Clinical review</Link>}</>} />
    <div className="surface px-5 py-5 sm:px-7"><ScreeningStepper active={step} /></div>
    {run && <PipelineStatusPanel run={run} />}
    {loading && <DataState label="Loading the integrated screening result" />}
    {error && <ErrorState message={error} />}
    {run?.status === 'FAILED' && <div className="rounded-2xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-900"><div className="flex items-start gap-3"><AlertTriangle size={18} className="mt-0.5" /><div><p className="font-extrabold">No clinical result was created</p><p className="mt-1 leading-6">{run.error?.message ?? run.message}</p><p className="mt-2 text-xs">Failed stage: {run.error?.stage ?? 'unknown'}. Resolve the configuration issue and start a new run.</p></div></div></div>}
    {run && run.status !== 'FAILED' && <>
      <TopSummary run={run} quality={quality} />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-6"><Viewer imageId={imageId} run={run} /><ClassificationPanel classification={classification} /><TrustPanel result={trust ?? null} /></div>
        <aside className="space-y-6"><SignalPanel run={run} /><QualityPanel quality={quality} /><EvidenceSummary run={run} /><div className="surface p-5"><div className="flex items-center gap-2"><Sparkles size={16} className="text-teal-600" /><p className="eyebrow">Explainability</p></div><p className="mt-2 text-sm font-bold text-ink">Verify why the model looked where it did.</p><p className="mt-1 text-xs leading-5 text-slate-500">Open the interactive Grad-CAM, lesion evidence, agreement, and stability view.</p><Link to="/screening/explain" state={{ imageId, screeningSessionId: run.screening_session_id, run }} className="btn-secondary mt-4 w-full">Open explainability <ArrowRightIcon /></Link></div></aside>
      </div>
    </>}
  </div>;
}

function TopSummary({ run, quality }: { run: ScreeningRun; quality?: { quality_decision?: string; quality_score?: number } }) {
  const classification = run.classification;
  const cards = [{ label: 'DR severity', value: classification?.predicted_grade_label ?? 'Not available', detail: classification ? `Level ${classification.predicted_grade} of 4` : 'No model output' }, { label: 'Referable DR', value: classification ? classification.referable_dr ? 'Yes' : 'No' : 'Not available', detail: classification ? `${Math.round(classification.referable_probability * 100)}% mapped probability` : 'No model output' }, { label: 'Trust score', value: run.retinaguard ? `${Math.round(run.retinaguard.trust_score * 100)}/100` : 'Not available', detail: run.retinaguard?.trust_category ?? 'Self-check not run' }, { label: 'Recommended action', value: actionLabel(run.triage?.recommendation), detail: quality?.quality_decision === 'UNGRADABLE' ? 'Image Trust Gate stopped the run' : 'Workflow recommendation only' }];
  return <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{cards.map((card, index) => <div key={card.label} className={`surface p-5 ${index === 0 ? 'border-teal-200 bg-teal-50/35' : ''}`}><p className="eyebrow">{card.label}</p><p className="section-title mt-3 truncate text-xl font-extrabold text-ink">{card.value}</p><p className="mt-1 truncate text-xs text-slate-500">{card.detail}</p></div>)}</div>;
}

function Viewer({ imageId, run }: { imageId?: string; run: ScreeningRun }) {
  const [visible, setVisible] = useState({ gradCam: true, lesions: true, evidence: false });
  const [opacity, setOpacity] = useState(0.72);
  const heatmap = run.explainability?.grad_cam.overlay_data_uri;
  const lesionMap = run.explainability?.lesion_evidence_map_data_uri;
  const evidenceMap = run.lesions?.evidence_map_data_uri;
  return <div className="surface overflow-hidden"><div className="flex flex-col justify-between gap-3 border-b border-line px-5 py-4 sm:flex-row sm:items-center"><div><p className="eyebrow">Fundus image viewer</p><h3 className="section-title mt-1 text-base font-extrabold">Original image and model-linked evidence</h3></div><span className="text-[11px] font-semibold text-slate-400">OD / OS · source image</span></div><div className="p-5"><div className="relative flex min-h-[310px] items-center justify-center overflow-hidden rounded-2xl bg-[#132643] sm:min-h-[460px]"><div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,#416c72_0%,#1a4156_40%,#10233f_75%)]" />{imageId ? <img src={imageContentUrl(imageId)} alt="Original fundus image" className="relative max-h-[460px] w-full object-contain" /> : <div className="relative px-8 text-center text-xs text-white/70">Original image preview is not available in this navigation state.</div>}{visible.gradCam && heatmap && <img src={heatmap} alt="Grad-CAM overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity }} />}{visible.lesions && lesionMap && <img src={lesionMap} alt="Lesion evidence overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity: opacity * 0.8 }} />}{visible.evidence && evidenceMap && <img src={evidenceMap} alt="Combined evidence overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity: opacity * 0.55 }} />}<span className="absolute left-3 top-3 rounded-lg bg-black/40 px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-widest text-white/80">Source + overlays</span></div><div className="mt-4 flex flex-wrap gap-2"><Toggle active={visible.gradCam} onClick={() => setVisible((current) => ({ ...current, gradCam: !current.gradCam }))} colour="bg-rose-400">Grad-CAM</Toggle><Toggle active={visible.lesions} onClick={() => setVisible((current) => ({ ...current, lesions: !current.lesions }))} colour="bg-amber-300">Lesion evidence</Toggle><Toggle active={visible.evidence} onClick={() => setVisible((current) => ({ ...current, evidence: !current.evidence }))} colour="bg-cyan-300">Other evidence</Toggle></div><label className="mt-4 flex items-center gap-3 text-xs text-slate-500"><span className="font-semibold">Overlay opacity</span><input aria-label="Overlay opacity" type="range" min="0" max="1" step="0.05" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} className="min-w-32 flex-1 accent-teal-600" /><span className="w-10 text-right font-bold">{Math.round(opacity * 100)}%</span></label><p className="mt-4 flex items-start gap-2 border-t border-line pt-4 text-[11px] leading-5 text-slate-400"><Info size={14} className="mt-0.5 shrink-0 text-teal-600" /> Overlays support review; they do not prove causality or replace clinical judgement.</p></div></div>;
}

function ClassificationPanel({ classification }: { classification: ScreeningRun['classification'] }) { if (!classification) return <div className="surface p-6"><p className="eyebrow">AI assessment</p><h3 className="section-title mt-2 text-lg font-extrabold">No classification output</h3><p className="mt-2 text-sm leading-6 text-slate-500">The configured model did not return a prediction. This is an intentional safe state.</p></div>; return <div className="surface p-6"><div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start"><div><p className="eyebrow">AI assessment</p><h3 className="section-title mt-1 text-2xl font-extrabold">{classification.predicted_grade_label}</h3><p className="mt-1 text-xs text-slate-500">Level {classification.predicted_grade} · {classification.model_name} · {classification.model_version}</p></div><StatusBadge tone={classification.referable_dr ? 'danger' : 'success'}>{classification.referable_dr ? 'Referable DR' : 'Non-referable'}</StatusBadge></div><div className="mt-5 space-y-3">{Object.entries(classification.probabilities).map(([label, value]) => <div key={label}><div className="flex justify-between text-xs font-semibold text-slate-500"><span>{label}</span><span>{Math.round(value * 100)}%</span></div><div className="mt-1.5 h-2 rounded-full bg-slate-100"><div className={`h-full rounded-full ${label === classification.predicted_grade_label ? 'bg-teal-500' : 'bg-slate-300'}`} style={{ width: `${value * 100}%` }} /></div></div>)}</div><p className="mt-5 border-t border-line pt-4 text-[11px] leading-5 text-slate-400">Raw confidence: {Math.round(classification.raw_confidence * 100)}%. Confidence is not a clinical trust guarantee. Referable mapping: grades {classification.referable_mapping.referable_grades.join(', ')}.</p></div>; }

function SignalPanel({ run }: { run: ScreeningRun }) { const factors = run.retinaguard?.contributing_factors ?? []; return <div className="surface p-5"><div className="flex items-center justify-between"><div><p className="eyebrow">Self-check signals</p><h3 className="section-title mt-1 text-base font-extrabold">RetinaGuard inputs</h3></div><StatusBadge tone={run.retinaguard ? 'teal' : 'neutral'}>{run.retinaguard ? 'Complete' : 'Unavailable'}</StatusBadge></div><div className="mt-5 space-y-3">{factors.slice(0, 8).map((factor) => <div key={factor.factor} className="flex items-center justify-between gap-3 border-b border-line pb-3 last:border-0 last:pb-0"><span className="text-xs capitalize text-slate-500">{factor.factor.replaceAll('_', ' ')}</span><span className="text-xs font-extrabold text-ink">{factor.raw_value == null ? 'Not run' : `${Math.round(factor.score * 100)}%`}</span></div>)}</div></div>; }
function QualityPanel({ quality }: { quality?: { quality_decision?: string; quality_score?: number; component_scores?: Record<string, number>; issues?: Array<{ type: string; message: string }> } }) { return <div className="surface p-5"><p className="eyebrow">Image quality</p><div className="mt-2 flex items-end justify-between"><p className="section-title text-3xl font-extrabold text-ink">{quality?.quality_score == null ? '—' : `${Math.round(quality.quality_score * 100)}%`}</p><StatusBadge tone={quality?.quality_decision === 'GRADABLE' ? 'success' : quality?.quality_decision === 'UNGRADABLE' ? 'danger' : 'warning'}>{quality?.quality_decision ?? 'Unavailable'}</StatusBadge></div><div className="mt-5 space-y-3">{Object.entries(quality?.component_scores ?? {}).map(([name, value]) => <div key={name}><div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-400"><span>{name.replaceAll('_', ' ')}</span><span>{Math.round(value * 100)}%</span></div><div className="mt-1 h-1.5 rounded-full bg-slate-100"><div className="h-full rounded-full bg-teal-500" style={{ width: `${value * 100}%` }} /></div></div>)}</div>{quality?.issues && quality.issues.length > 0 && <p className="mt-4 text-xs leading-5 text-amber-700">{quality.issues.length} quality issue(s) recorded.</p>}</div>; }
function PipelineStatusPanel({ run }: { run: ScreeningRun }) {
  const rows = [
    { label: 'Primary screening', state: run.primary_status ?? (run.classification ? 'COMPLETED' : run.status), detail: 'Quality, classification, RetinaGuard and triage' },
    { label: 'DR classification', state: run.classification ? 'COMPLETED' : run.quality?.final?.quality_decision === 'UNGRADABLE' ? 'NOT_RUN' : 'PENDING', detail: run.classification?.model_version ?? 'No model output' },
    { label: 'RetinaGuard', state: run.retinaguard ? 'COMPLETED' : 'NOT_RUN', detail: run.retinaguard?.trust_category ?? 'No self-check output' },
    { label: 'Triage', state: run.triage ? 'COMPLETED' : 'PENDING', detail: run.triage?.recommendation ?? 'Pending' },
    { label: 'Lesion / vessel evidence', state: evidenceStageState(run, ['retinal_structure_analysis', 'lesion_detection']), detail: 'Optional supporting evidence' },
    { label: 'Grad-CAM / agreement', state: evidenceStageState(run, ['grad_cam', 'attention_lesion_agreement']), detail: 'Optional explainability evidence' },
  ];
  const evidenceLabel = run.evidence_status === 'AVAILABLE' ? 'Evidence available' : run.evidence_status === 'PROCESSING' ? 'Evidence processing' : run.evidence_status === 'TIMED_OUT' ? 'Evidence timed out' : run.evidence_status === 'UNAVAILABLE' ? 'Evidence unavailable' : run.evidence_status ?? 'Not run';
  return <div className="surface p-5"><div className="flex flex-col justify-between gap-2 sm:flex-row sm:items-start"><div><p className="eyebrow">Pipeline status</p><h3 className="section-title mt-1 text-base font-extrabold">Primary result and optional evidence</h3></div><StatusBadge tone={run.evidence_status === 'AVAILABLE' ? 'success' : run.evidence_status === 'TIMED_OUT' || run.evidence_status === 'UNAVAILABLE' ? 'warning' : 'teal'}>{evidenceLabel}</StatusBadge></div><div className="mt-5 grid gap-2 md:grid-cols-2 xl:grid-cols-3">{rows.map((row) => <div key={row.label} className="flex items-start gap-3 rounded-xl border border-line bg-mist/60 p-3"><StageIcon state={row.state} /><div className="min-w-0"><p className="text-xs font-extrabold text-ink">{row.label}</p><p className="mt-1 text-[11px] leading-4 text-slate-500">{stageLabel(row.state)} · {row.detail}</p></div></div>)}</div>{run.evidence_status && run.evidence_status !== 'AVAILABLE' && run.classification && <p className="mt-4 text-[11px] leading-5 text-slate-500">Optional evidence does not block the primary screening result and is never treated as negative evidence when unavailable.</p>}</div>;
}
function evidenceStageState(run: ScreeningRun, stages: string[]) { const values = stages.map((stage) => run.stage_status?.[stage] ?? 'PENDING'); if (values.some((value) => value === 'PROCESSING' || value === 'QUEUED' || value === 'PENDING')) return 'PROCESSING'; if (values.some((value) => value === 'TIMED_OUT')) return 'TIMED_OUT'; if (values.some((value) => value === 'UNAVAILABLE' || value === 'FAILED')) return 'UNAVAILABLE'; return 'COMPLETED'; }
function stageLabel(state: string) { return ({ COMPLETED: 'Available', PROCESSING: 'Processing', QUEUED: 'Queued', PENDING: 'Pending', TIMED_OUT: 'Not completed within runtime budget', UNAVAILABLE: 'Unavailable', NOT_RUN: 'Not run', QUALITY_BLOCKED: 'Blocked by quality gate', FAILED: 'Failed safely' } as Record<string, string>)[state] ?? state; }
function StageIcon({ state }: { state: string }) { if (state === 'COMPLETED') return <CheckCircle2 size={17} className="mt-0.5 shrink-0 text-teal-600" aria-label="Available" />; if (state === 'PROCESSING' || state === 'QUEUED') return <Clock3 size={17} className="mt-0.5 shrink-0 text-amber-600" aria-label="Processing" />; if (state === 'TIMED_OUT' || state === 'UNAVAILABLE' || state === 'FAILED') return <XCircle size={17} className="mt-0.5 shrink-0 text-amber-700" aria-label={stageLabel(state)} />; return <AlertTriangle size={17} className="mt-0.5 shrink-0 text-slate-400" aria-label={stageLabel(state)} />; }
function EvidenceSummary({ run }: { run: ScreeningRun }) { const modules = Object.values(run.lesions?.modules ?? {}); return <div className="surface p-5"><p className="eyebrow">Clinical evidence</p><h3 className="section-title mt-1 text-base font-extrabold">Supporting findings</h3><div className="mt-4 space-y-2">{modules.length === 0 ? <p className="text-xs leading-5 text-slate-500">No lesion module output was available for this run.</p> : modules.map((module) => <div key={module.module} className="flex items-center justify-between gap-3 rounded-xl bg-mist p-3"><span className="truncate text-xs font-bold capitalize text-ink">{module.module.replaceAll('_', ' ')}</span><span className="text-xs font-extrabold text-slate-600">{module.count ?? module.status}</span></div>)}</div>{run.evidence_message && run.evidence_status !== 'AVAILABLE' && <p className="mt-4 border-t border-line pt-4 text-[11px] leading-5 text-slate-500">{run.evidence_message}</p>}</div>; }
function Toggle({ active, onClick, colour, children }: { active: boolean; onClick: () => void; colour: string; children: React.ReactNode }) { return <button onClick={onClick} className={`inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-[11px] font-bold ${active ? 'border-line bg-white text-ink' : 'border-transparent bg-mist text-slate-400'}`}><span className={`h-2 w-2 rounded-full ${colour} ${active ? '' : 'opacity-35'}`} />{children}</button>; }
function actionLabel(value?: string) { return ({ AI_TRIAGE_MAY_PROCEED: 'AI triage may proceed', SPECIALIST_REVIEW_RECOMMENDED: 'Specialist review', HUMAN_REVIEW_REQUIRED: 'Human review', RECAPTURE_OR_SPECIALIST_REVIEW: 'Recapture / review', RECAPTURE_IMAGE: 'Recapture image' } as Record<string, string>)[value ?? ''] ?? 'Pending'; }
function ArrowRightIcon() { return <span aria-hidden="true">→</span>; }
