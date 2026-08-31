import { AlertTriangle, ArrowRight, CheckCircle2, CircleHelp, Eye, FlaskConical, ShieldAlert } from 'lucide-react';
import { useEffect, useState } from 'react';
import { DataState, ErrorState } from '../components/DataState';
import { PageIntro } from '../components/PageIntro';
import { StatusBadge } from '../components/StatusBadge';
import { getDemoScenarios, runDemoScenario, type DemoScenario } from '../services/api';

export function DemoPage() {
  const [scenarios, setScenarios] = useState<DemoScenario[]>([]);
  const [selected, setSelected] = useState<DemoScenario | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDemoScenarios().then((payload) => setScenarios(payload.scenarios)).catch((requestError) => setError(requestError instanceof Error ? requestError.message : 'Demo mode is unavailable.')).finally(() => setLoading(false));
  }, []);

  async function openScenario(scenario: DemoScenario) {
    setError('');
    try {
      const result = await runDemoScenario(scenario.scenario_id);
      setSelected(result.scenario);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Unable to run demo scenario.');
    }
  }

  if (loading) return <DataState label="Loading controlled demo scenarios" />;
  if (error && scenarios.length === 0) return <div className="space-y-6"><PageIntro eyebrow="Sandbox" title="Demo mode" description="Controlled product walkthroughs are available only when explicitly enabled for development or testing." /><ErrorState message={error} /></div>;

  return <div className="space-y-7">
    <PageIntro eyebrow="Sandbox · synthetic data" title="RETINA-NEXUS demo mode" description="Walk through the three decision paths without creating patient records or claiming clinical performance." action={<StatusBadge tone="warning" dot>Demo only</StatusBadge>} />
    <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-900"><div className="flex items-start gap-3"><FlaskConical size={17} className="mt-0.5 shrink-0" /><p><strong>Controlled sample workflow.</strong> These scenarios are synthetic fixtures. They are not patient images, validation data, or diagnostic results, and they are never used as a production fallback.</p></div></div>
    <div className="grid gap-5 lg:grid-cols-3">{scenarios.map((scenario) => <ScenarioCard key={scenario.scenario_id} scenario={scenario} active={selected?.scenario_id === scenario.scenario_id} onRun={() => openScenario(scenario)} />)}</div>
    {selected && <DemoResult scenario={selected} />}
  </div>;
}

function ScenarioCard({ scenario, active, onRun }: { scenario: DemoScenario; active: boolean; onRun: () => void }) {
  const category = scenario.expected_category ?? 'UNGRADABLE';
  const tone = category === 'TRUSTED' ? 'success' : category === 'UNRELIABLE' || category === 'UNGRADABLE' ? 'danger' : 'warning';
  return <button onClick={onRun} className={`surface w-full text-left transition hover:-translate-y-0.5 hover:shadow-lg ${active ? 'border-teal-400 ring-4 ring-teal-50' : ''}`}><div className="border-b border-line p-5"><div className="flex items-center justify-between gap-3"><span className="eyebrow text-teal-700">{scenario.image_label}</span><StatusBadge tone={tone}>{category}</StatusBadge></div><h2 className="section-title mt-4 text-base font-extrabold">{scenario.title}</h2><p className="mt-2 text-xs leading-5 text-slate-500">{scenario.summary}</p></div><div className="flex items-center justify-between p-5 text-xs font-extrabold text-teal-700"><span>Expected: {scenario.expected_action}</span><ArrowRight size={16} /></div></button>;
}

function DemoResult({ scenario }: { scenario: DemoScenario }) {
  const quality = scenario.quality ?? {};
  const classification = scenario.classification ?? {};
  const guard = scenario.retinaguard ?? {};
  const triage = scenario.triage ?? {};
  const isBlocked = quality.quality_decision === 'UNGRADABLE';
  return <section className="surface overflow-hidden"><div className="flex flex-col justify-between gap-4 border-b border-line bg-mist/50 px-6 py-5 sm:flex-row sm:items-center"><div><p className="eyebrow">{scenario.image_label} · simulated run</p><h2 className="section-title mt-1 text-xl font-extrabold">{scenario.title}</h2></div><StatusBadge tone={isBlocked ? 'danger' : guard.trust_category === 'TRUSTED' ? 'success' : 'warning'} dot>{isBlocked ? 'UNGRADABLE' : guard.trust_category}</StatusBadge></div><div className="grid gap-5 p-6 sm:grid-cols-2 xl:grid-cols-4"><ResultMetric icon={Eye} label="Image quality" value={quality.quality_decision ?? 'Unavailable'} detail={`${Math.round((quality.quality_score ?? 0) * 100)}%${quality.enhancement_applied ? ` · enhanced${quality.recheck_score ? ` · recheck ${Math.round(quality.recheck_score * 100)}%` : ''}` : ''}`} /><ResultMetric icon={CircleHelp} label="DR assessment" value={classification.predicted_grade_label ?? 'Not started'} detail={classification.referable_dr == null ? 'Clinical AI blocked' : `${classification.referable_dr ? 'Referable' : 'Non-referable'} · ${Math.round((classification.raw_confidence ?? 0) * 100)}% raw confidence`} /><ResultMetric icon={ShieldAlert} label="RetinaGuard" value={guard.trust_category ?? 'Not run'} detail={guard.trust_score == null ? 'Quality gate stopped downstream AI' : `Trust score ${Math.round(guard.trust_score * 100)}%`} /><ResultMetric icon={isBlocked ? AlertTriangle : CheckCircle2} label="Next action" value={triage.display_action ?? 'Review'} detail={triage.recommendation ?? 'No recommendation'} /></div>{!isBlocked && <div className="grid gap-4 border-t border-line px-6 py-5 md:grid-cols-2"><div className="rounded-xl bg-teal-50 p-4 text-xs leading-5 text-teal-900"><p className="font-extrabold">Evidence link</p><p className="mt-1">{scenario.lesions?.summary?.length ?? 0} synthetic lesion summaries · {scenario.explainability?.attention_lesion_agreement?.status ?? 'No agreement result'}.</p></div><div className="rounded-xl bg-slate-50 p-4 text-xs leading-5 text-slate-600"><p className="font-extrabold text-ink">Demo boundary</p><p className="mt-1">The displayed confidence and trust values are fixture values for interaction testing, not measured model performance.</p></div></div>}</section>;
}

function ResultMetric({ icon: Icon, label, value, detail }: { icon: typeof Eye; label: string; value: string; detail: string }) { return <div className="rounded-xl border border-line p-4"><Icon size={17} className="text-teal-600" /><p className="mt-4 text-[10px] font-bold uppercase tracking-wider text-slate-400">{label}</p><p className="mt-1 text-sm font-extrabold text-ink">{value}</p><p className="mt-1 text-[11px] leading-5 text-slate-500">{detail}</p></div>; }
