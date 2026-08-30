import { Bell, Menu, Search, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import { useLocation } from 'react-router-dom';

const titles: Record<string, { title: string; crumb: string }> = {
  '/': { title: 'Good morning', crumb: 'Dashboard' },
  '/screening/new': { title: 'Start a new screening', crumb: 'New screening' },
  '/screening/results': { title: 'Screening result', crumb: 'New screening / Result' },
  '/screening/explain': { title: 'Explainability', crumb: 'Screening / Evidence verification' },
  '/history': { title: 'Screening history', crumb: 'Screening history' },
  '/patients': { title: 'Patient directory', crumb: 'Patients' },
  '/review': { title: 'Clinical review', crumb: 'Clinical review' },
  '/reports': { title: 'Reports', crumb: 'Reports' },
  '/analytics': { title: 'Care intelligence', crumb: 'Analytics' },
  '/monitoring': { title: 'Model monitoring', crumb: 'Model monitoring' },
  '/settings': { title: 'Workspace settings', crumb: 'Settings' },
  '/datasets': { title: 'Dataset management', crumb: 'Admin / Data governance' },
  '/demo': { title: 'Controlled demo', crumb: 'Verification' },
};

export function Header({ onMenu }: { onMenu: () => void }) {
  const { pathname } = useLocation();
  const copy = titles[pathname] ?? titles['/'];
  return <header className="flex min-h-[82px] items-center justify-between gap-4 border-b border-line bg-white px-4 py-4 sm:px-8 lg:px-10">
    <div className="flex min-w-0 items-center gap-3"><button aria-label="Open navigation" onClick={onMenu} className="rounded-lg p-2 text-slate-500 hover:bg-mist lg:hidden"><Menu size={20} /></button><div className="min-w-0"><div className="hidden items-center gap-2 text-[11px] font-semibold text-slate-400 sm:flex"><span>RETINA-NEXUS</span><span>/</span><span className="text-teal-600">{copy.crumb}</span></div><h1 className="section-title mt-1 truncate text-lg font-extrabold text-ink sm:text-[21px]">{copy.title}</h1></div></div>
    <div className="flex items-center gap-2 sm:gap-3"><div className="hidden items-center gap-2 rounded-xl border border-line bg-mist px-3 py-2 text-xs text-slate-400 md:flex"><Search size={15} /><span>Search screenings</span><kbd className="ml-5 rounded bg-white px-1.5 py-0.5 text-[10px] text-slate-400">Cmd K</kbd></div><button aria-label="Filter" className="hidden rounded-xl border border-line p-2.5 text-slate-500 hover:border-teal-300 hover:text-teal-600 sm:block"><SlidersHorizontal size={17} /></button><button aria-label="Notifications" className="relative rounded-xl border border-line p-2.5 text-slate-500 hover:border-teal-300 hover:text-teal-600"><Bell size={17} /><span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-rose-400 ring-2 ring-white" /></button><div className="hidden items-center gap-2 rounded-xl border border-line px-2.5 py-2 sm:flex"><ShieldCheck size={15} className="text-teal-600" /><span className="text-[11px] font-bold text-slate-600">Clinical workspace</span></div><div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#dce9ef] text-xs font-extrabold text-ink">CT</div></div>
  </header>;
}
