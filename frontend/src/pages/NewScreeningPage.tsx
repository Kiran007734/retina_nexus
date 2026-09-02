import { AlertTriangle, ArrowRight, Check, FileCheck2, Info, LoaderCircle, RefreshCw, ShieldCheck, UserRound } from 'lucide-react';
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { DataState } from '../components/DataState';
import { PageIntro } from '../components/PageIntro';
import { ScreeningStepper } from '../components/ScreeningStepper';
import { StatusBadge } from '../components/StatusBadge';
import { UploadArea } from '../components/UploadArea';
import { ApiRequestError, createPatient, getPatients, runScreening, uploadFundusImage, type ApiErrorCategory, type ScreeningRun } from '../services/api';

type FlowState = 'idle' | 'uploading' | 'processing' | 'complete' | 'error';

export function NewScreeningPage() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<File | null>(null);
  const [patientId, setPatientId] = useState('');
  const [eye, setEye] = useState<'left' | 'right'>('right');
  const [state, setState] = useState<FlowState>('idle');
  const [imageId, setImageId] = useState<string | null>(null);
  const [run, setRun] = useState<ScreeningRun | null>(null);
  const [error, setError] = useState('');
  const [errorCategory, setErrorCategory] = useState<ApiErrorCategory | null>(null);

  async function startScreening() {
    const identifier = patientId.trim();
    if (!selected || !identifier) return;
    if (identifier.length < 3) {
      setState('error');
      setErrorCategory('REQUEST_VALIDATION_FAILURE');
      setError('Patient identifier must be at least 3 characters.');
      return;
    }
    setError(''); setErrorCategory(null); setRun(null); setState('uploading');
    try {
      let patient: { id: string };
      try { patient = await createPatient({ anonymized_identifier: identifier }); } catch (createError) {
        if (!(createError instanceof ApiRequestError) || createError.status !== 409) throw createError;
        const existing = await getPatients(); const match = existing.find((item) => item.anonymized_identifier === identifier); if (!match) throw createError; patient = match;
      }
      const uploaded = await uploadFundusImage(patient.id, eye, selected); setImageId(uploaded.image_id); setState('processing');
      const result = await runScreening(uploaded.image_id); setRun(result); setState('complete');
    } catch (requestError) {
      setState('error');
      if (requestError instanceof ApiRequestError) {
        setErrorCategory(requestError.category);
        const details = requestError.validationErrors.map((item) => `${item.loc.join('.')} — ${item.msg}`).join(' ');
        setError(details ? `${requestError.message} ${details}` : requestError.message);
      } else {
        setErrorCategory('INTERNAL_SERVER_ERROR');
        setError(requestError instanceof Error ? requestError.message : 'The screening pipeline could not process this image.');
      }
    }
  }

  const quality = run?.quality?.final as { quality_decision?: string; quality_score?: number; recommended_action?: string; issues?: Array<{ type: string; message: string; recommendation: string }> } | undefined;
  const blocked = quality?.quality_decision === 'UNGRADABLE';
  const progress = run ? runProgress(run) : state === 'processing' ? 2 : 0;
  return <div className="space-y-7">
    <PageIntro eyebrow="Guided workflow" title="Start a new screening" description="Move from trusted image intake to an evidence-linked, self-checking result in one controlled flow." action={<div className="flex flex-wrap gap-2"><Link to="/demo" className="btn-secondary">Explore demo <ArrowRight size={15} /></Link><Link to="/history" className="btn-secondary">View history <ArrowRight size={15} /></Link></div>} />
    <div className="surface overflow-hidden px-5 py-5 sm:px-7"><ScreeningStepper active={progress} /></div>
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_330px]">
      <div className="space-y-5">
        <section className="surface p-6 sm:p-7"><div className="flex items-start gap-3"><div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-50 text-teal-700"><UserRound size={18} /></div><div><p className="eyebrow">Step 1 · Patient context</p><h2 className="section-title mt-1 text-lg font-extrabold">Keep the record anonymous</h2><p className="mt-1 text-xs leading-5 text-slate-500">Use the identifier your care team already uses. No direct identifiers are required.</p></div></div><div className="mt-6 grid gap-4 sm:grid-cols-2"><label><span className="mb-2 block text-xs font-bold text-ink">Patient identifier</span><input value={patientId} onChange={(event) => setPatientId(event.target.value)} disabled={state === 'uploading' || state === 'processing'} className="w-full rounded-xl border border-line px-3.5 py-3 text-sm font-semibold text-ink outline-none focus:border-teal-400 focus:ring-4 focus:ring-teal-50 disabled:bg-mist" /></label><label><span className="mb-2 block text-xs font-bold text-ink">Eye being screened</span><select value={eye} onChange={(event) => setEye(event.target.value as 'left' | 'right')} disabled={state === 'uploading' || state === 'processing'} className="w-full rounded-xl border border-line bg-white px-3.5 py-3 text-sm font-semibold text-ink outline-none focus:border-teal-400 focus:ring-4 focus:ring-teal-50 disabled:bg-mist"><option value="right">OD · Right eye</option><option value="left">OS · Left eye</option></select></label></div></section>
        <section className="surface p-6 sm:p-7"><div className="flex items-start gap-3"><div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-50 text-teal-700"><FileCheck2 size={18} /></div><div><p className="eyebrow">Step 1 · Fundus image</p><h2 className="section-title mt-1 text-lg font-extrabold">Upload a clear retinal image</h2><p className="mt-1 text-xs leading-5 text-slate-500">JPEG or PNG · the Image Trust Gate checks integrity, dimensions, channels, and gradability.</p></div></div><div className="mt-6"><UploadArea onFile={(file) => { setSelected(file); setRun(null); setImageId(null); setError(''); setState('idle'); }} /></div></section>
        {(state === 'uploading' || state === 'processing') && <div className="surface overflow-hidden border-teal-100"><div className="flex items-center gap-3 bg-teal-50 px-5 py-4"><div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white text-teal-600 shadow-sm"><LoaderCircle size={19} className="animate-spin" /></div><div><p className="text-sm font-extrabold text-ink">{state === 'uploading' ? 'Securing the image' : 'Running the screening pipeline'}</p><p className="mt-1 text-xs text-slate-500">Each stage records its status; no result is inferred if a module fails.</p></div></div><div className="grid gap-2 px-5 py-4 sm:grid-cols-2">{['Validate image', 'Assess quality', 'Classify DR', 'Analyze evidence', 'Run RetinaGuard', 'Prepare triage'].map((item, index) => <div key={item} className="flex items-center gap-2 text-xs text-slate-500"><LoaderCircle size={13} className={index <= progress ? 'animate-spin text-teal-500' : 'text-slate-300'} />{item}</div>)}</div></div>}
        {state === 'error' && <div role="alert" className="rounded-2xl border border-rose-200 bg-rose-50 p-5 text-sm text-rose-800"><div className="flex items-start gap-3"><AlertTriangle size={18} className="mt-0.5 shrink-0" /><div><p className="font-extrabold">Screening could not be completed</p><p className="mt-1 leading-6">{error}</p><p className="mt-2 text-xs text-rose-700">{failureGuidance(errorCategory)}</p></div></div></div>}
        {run && <RunSummary run={run} imageId={imageId} />}
        <div className="flex flex-col-reverse justify-between gap-3 sm:flex-row sm:items-center"><p className="flex items-center gap-2 text-[11px] leading-5 text-slate-400"><Info size={14} className="shrink-0 text-teal-600" /> Ungradable images stop before clinical AI and receive recapture guidance.</p>{run && imageId ? <button onClick={() => navigate('/screening/results', { state: { screeningId: run.screening_id, imageId, run } })} className="btn-primary">Open screening result <ArrowRight size={16} /></button> : <button disabled={!selected || !patientId.trim() || state === 'uploading' || state === 'processing'} onClick={startScreening} className="btn-primary disabled:cursor-not-allowed disabled:opacity-40">Run secure screening <ArrowRight size={16} /></button>}</div>
      </div>
      <aside className="space-y-4"><div className="surface p-5"><p className="eyebrow">What the workflow protects</p><div className="mt-5 space-y-4">{[['01', 'Quality before AI', 'The Trust Gate blocks ungradable inputs.'], ['02', 'Evidence alongside grade', 'Structures and lesions remain separate supporting evidence.'], ['03', 'Human review when needed', 'RetinaGuard makes uncertainty visible to the care team.']].map(([number, title, detail]) => <div key={number} className="flex gap-3"><span className="text-[11px] font-extrabold text-teal-600">{number}</span><div><p className="text-xs font-extrabold text-ink">{title}</p><p className="mt-1 text-xs leading-5 text-slate-500">{detail}</p></div></div>)}</div></div><div className="rounded-2xl border border-ink/10 bg-ink p-5 text-white"><div className="flex items-center gap-2"><ShieldCheck size={17} className="text-teal-300" /><p className="text-xs font-extrabold">Trust-first care workflow</p></div><p className="mt-3 text-xs leading-5 text-slate-300">AI output is a screening recommendation. A clinician remains responsible for the final decision.</p></div></aside>
    </div>
  </div>;
}

function RunSummary({ run, imageId }: { run: ScreeningRun; imageId: string | null }) {
  const quality = run.quality?.final as { quality_decision?: string; quality_score?: number; recommended_action?: string; issues?: Array<{ type: string; message: string; recommendation: string }> } | undefined;
  const blocked = quality?.quality_decision === 'UNGRADABLE';
  return <div className={`rounded-2xl border p-5 ${run.status === 'FAILED' ? 'border-rose-200 bg-rose-50' : blocked ? 'border-amber-200 bg-amber-50' : 'border-teal-100 bg-teal-50/60'}`}><div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start"><div><p className="eyebrow">Pipeline result</p><h2 className="section-title mt-1 text-lg font-extrabold">{run.status === 'FAILED' ? 'Run failed safely' : blocked ? 'Recapture recommended' : 'Screening pipeline complete'}</h2><p className="mt-1 text-xs leading-5 text-slate-600">{run.message}</p></div><StatusBadge tone={run.status === 'FAILED' ? 'danger' : blocked ? 'warning' : 'success'}>{run.status}</StatusBadge></div>{run.error && <p className="mt-4 rounded-xl bg-white/70 p-3 text-xs leading-5 text-rose-800">Stage <strong>{run.error.stage}</strong>: {run.error.message}</p>}{quality && <div className="mt-4 grid gap-3 sm:grid-cols-3"><Mini label="Quality" value={`${quality.quality_decision ?? '—'} · ${Math.round((quality.quality_score ?? 0) * 100)}%`} /><Mini label="Image" value={imageId ? 'Stored securely' : 'Unavailable'} /><Mini label="Next action" value={blocked ? 'Recapture image' : run.triage?.recommendation ?? 'Review result'} /></div>}{blocked && quality?.issues && quality.issues.length > 0 && <div className="mt-4 space-y-2">{quality.issues.map((issue) => <div key={issue.type} className="rounded-xl bg-white/70 p-3 text-xs"><p className="font-extrabold text-ink">{issue.type.replaceAll('_', ' ')}</p><p className="mt-1 text-slate-600">{issue.message}</p><p className="mt-1 font-semibold text-teal-800">{issue.recommendation}</p></div>)}</div>}</div>;
}

function Mini({ label, value }: { label: string; value: string }) { return <div className="rounded-xl bg-white/70 p-3"><p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{label}</p><p className="mt-1 truncate text-xs font-extrabold text-ink">{value}</p></div>; }
function runProgress(run: ScreeningRun) { const completed = Object.values(run.stage_status).filter((value) => value === 'COMPLETED').length; return Math.min(5, Math.max(1, completed)); }
function failureGuidance(category: ApiErrorCategory | null) {
  switch (category) {
    case 'REQUEST_VALIDATION_FAILURE': return 'REQUEST_VALIDATION_FAILURE · The request was rejected before image processing. Correct the input and retry.';
    case 'API_CONNECTION_FAILURE': return 'API_CONNECTION_FAILURE · No clinical prediction was created because the backend could not be reached.';
    case 'MODEL_UNAVAILABLE': return 'MODEL_UNAVAILABLE · The API responded, but a registered model artifact was unavailable; no prediction was created.';
    case 'INFERENCE_FAILURE': return 'INFERENCE_FAILURE · The request reached the pipeline, but inference failed safely; no prediction was created.';
    case 'QUALITY_GATE_REJECTION': return 'QUALITY_GATE_REJECTION · The Image Trust Gate rejected this input; follow the recapture guidance before continuing.';
    case 'INTERNAL_SERVER_ERROR': return 'INTERNAL_SERVER_ERROR · The backend reported an internal failure before a result could be rendered.';
    default: return 'No clinical prediction was created. Review the request status and retry when the underlying issue is resolved.';
  }
}
