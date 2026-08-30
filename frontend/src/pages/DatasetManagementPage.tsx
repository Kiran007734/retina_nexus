import { AlertTriangle, BarChart3, Database, FileCheck2, GitBranch, RefreshCw, ShieldCheck } from 'lucide-react';
import { useEffect, useState } from 'react';

import { DataState, EmptyState, ErrorState } from '../components/DataState';
import { StatusBadge } from '../components/StatusBadge';
import { getDatasetStatistics, getDatasets } from '../services/api';
import type { DatasetRecord, DatasetStatisticsRecord, StatusTone } from '../types';

const statusCopy: Record<DatasetRecord['status'], { label: string; tone: StatusTone }> = {
  not_acquired: { label: 'Not acquired', tone: 'neutral' },
  available: { label: 'Available', tone: 'teal' },
  validating: { label: 'Validating', tone: 'warning' },
  ready: { label: 'Ready', tone: 'success' },
  blocked: { label: 'Blocked', tone: 'danger' },
};
const availabilityCopy: Record<string, { label: string; tone: StatusTone }> = {
  AVAILABLE: { label: 'Available', tone: 'success' },
  'PARTIALLY AVAILABLE': { label: 'Partially available', tone: 'warning' },
  MISSING: { label: 'Missing', tone: 'neutral' },
  INVALID: { label: 'Invalid', tone: 'danger' },
};

export function DatasetManagementPage() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [statistics, setStatistics] = useState<Record<string, DatasetStatisticsRecord>>({});
  const [live, setLive] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const items = await getDatasets();
      const nextStatistics: Record<string, DatasetStatisticsRecord> = {};
      await Promise.all(items.map(async (item) => {
        if (!item.id) return;
        try { nextStatistics[item.id] = await getDatasetStatistics(item.id); } catch { /* Registry rows remain usable without optional statistics. */ }
      }));
      setDatasets(items);
      setStatistics(nextStatistics);
      setLive(true);
    } catch (reason: unknown) {
      setLive(false);
      setError(reason instanceof Error ? reason.message : 'Dataset registry could not be loaded.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void load(); }, []);
  const statRows = Object.values(statistics);
  const totalFiles = statRows.reduce((sum, item) => sum + item.total_files, 0);
  const readableFiles = statRows.reduce((sum, item) => sum + item.readable_files, 0);
  const corruptedFiles = statRows.reduce((sum, item) => sum + item.corrupted_files, 0);
  const duplicateFiles = statRows.reduce((sum, item) => sum + item.duplicate_exact_count + item.duplicate_perceptual_count, 0);
  const hasStatistics = statRows.length > 0;

  return <div className="space-y-6">
    <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end"><div><div className="flex flex-wrap items-center gap-2"><p className="text-sm text-slate-500">Admin · Data governance</p><StatusBadge tone={live ? 'success' : 'neutral'} dot>{live ? 'Registry connected' : 'Registry unavailable'}</StatusBadge></div><p className="mt-1 text-sm text-slate-500">Track source readiness before any dataset is used for training or evaluation.</p></div><button className="btn-secondary" onClick={() => void load()} disabled={loading}><RefreshCw size={15} /> Refresh registry</button></div>
    <div className="rounded-2xl border border-teal-100 bg-teal-50 px-4 py-3 text-xs text-teal-800"><div className="flex items-start gap-2"><ShieldCheck size={16} className="mt-0.5 shrink-0" /><p><strong>Readiness is an engineering signal.</strong> It combines file, label, duplicate and split checks; it is not a clinical validation score. Dataset content is intentionally not bundled.</p></div></div>
    {loading && <DataState label="Loading dataset registry" />}
    {error && <ErrorState message={error} onRetry={() => void load()} />}
    {!loading && !error && datasets.length === 0 && <EmptyState title="No datasets are registered" detail="Dataset registry entries will appear after the API database is initialized." />}
    {!loading && !error && datasets.length > 0 && <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">{datasets.map((dataset) => <DatasetCard key={dataset.slug} dataset={dataset} statistics={dataset.id ? statistics[dataset.id] : undefined} />)}</div>}
    <div className="grid gap-5 lg:grid-cols-[1fr_340px]"><div className="card p-5"><div className="flex items-center gap-2"><div className="rounded-lg bg-teal-50 p-2 text-teal-600"><BarChart3 size={17} /></div><div><h2 className="section-title text-base font-extrabold">Governance indicators</h2><p className="mt-1 text-xs text-slate-400">Latest statistics returned by the registry</p></div></div><div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"><Indicator icon={FileCheck2} label="Readable files" value={hasStatistics ? `${readableFiles.toLocaleString()} / ${totalFiles.toLocaleString()}` : 'Not run'} /><Indicator icon={GitBranch} label="Split integrity" value="See validation" /><Indicator icon={AlertTriangle} label="Duplicates" value={hasStatistics ? duplicateFiles.toLocaleString() : 'Not run'} /><Indicator icon={Database} label="Corrupted files" value={hasStatistics ? corruptedFiles.toLocaleString() : 'Not run'} /></div></div><div className="card p-5"><p className="eyebrow">Next action</p><h2 className="section-title mt-2 text-base font-extrabold">Run a validation pass</h2><p className="mt-2 text-xs leading-5 text-slate-500">Place an authorized dataset in its raw folder, then run the governance CLI to populate these indicators.</p><code className="mt-4 block rounded-xl bg-ink p-3 text-[11px] text-teal-100">python scripts/validate_dataset.py<br />--dataset aptos2019</code></div></div>
  </div>;
}

function DatasetCard({ dataset, statistics }: { dataset: DatasetRecord; statistics?: DatasetStatisticsRecord }) {
  const status = availabilityCopy[dataset.availability_status ?? ''] ?? statusCopy[dataset.status];
  const duplicateCount = statistics ? statistics.duplicate_exact_count + statistics.duplicate_perceptual_count : null;
  const classCount = statistics?.class_distribution ? Object.keys(statistics.class_distribution).length : null;
  return <div className="card p-5 transition hover:-translate-y-0.5 hover:shadow-lift"><div className="flex items-start justify-between gap-3"><div className="rounded-xl bg-teal-50 p-2.5 text-teal-600"><Database size={18} /></div><StatusBadge tone={status.tone}>{status.label}</StatusBadge></div><h2 className="section-title mt-5 min-h-10 text-sm font-extrabold leading-5 text-ink">{dataset.name}</h2><p className="mt-1 min-h-8 text-xs leading-4 text-slate-400">{dataset.purpose}</p><div className="mt-5 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-line pt-4 text-xs"><Stat label="Images" value={dataset.image_count == null ? '—' : dataset.image_count.toLocaleString()} /><Stat label="Readiness" value={dataset.readiness_score == null ? '—' : `${dataset.readiness_score}/100`} /><Stat label="Duplicates" value={duplicateCount == null ? 'Not run' : duplicateCount.toLocaleString()} /><Stat label="Corrupted" value={statistics == null ? 'Not run' : statistics.corrupted_files.toLocaleString()} /><Stat label="Classes" value={classCount == null ? 'Not analyzed' : classCount.toString()} /><Stat label="Splits" value="See validation" /></div></div>;
}

function Stat({ label, value }: { label: string; value: string }) { return <div><p className="text-[10px] text-slate-400">{label}</p><p className="mt-1 truncate text-xs font-bold text-ink">{value}</p></div>; }
function Indicator({ icon: Icon, label, value }: { icon: typeof Database; label: string; value: string }) { return <div className="rounded-xl bg-mist p-3"><Icon size={15} className="text-teal-600" /><p className="mt-3 text-[11px] font-bold text-slate-500">{label}</p><p className="mt-1 text-xs font-extrabold text-ink">{value}</p></div>; }
