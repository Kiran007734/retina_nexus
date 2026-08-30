import { Activity, BarChart3, ClipboardCheck, Database, FileText, History, LayoutDashboard, LogOut, Settings, Upload, X } from 'lucide-react';
import { NavLink, useNavigate } from 'react-router-dom';
import { BrandMark } from './BrandMark';

const primaryNav = [
  { label: 'Dashboard', to: '/', icon: LayoutDashboard },
  { label: 'New screening', to: '/screening/new', icon: Upload },
  { label: 'Screening history', to: '/history', icon: History },
  { label: 'Clinical review', to: '/review', icon: ClipboardCheck },
  { label: 'Reports', to: '/reports', icon: FileText },
  { label: 'Analytics', to: '/analytics', icon: BarChart3 },
  { label: 'Model monitoring', to: '/monitoring', icon: Activity },
  { label: 'Settings', to: '/settings', icon: Settings },
];

export function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate();
  return <>
    {open && <button aria-label="Close navigation" onClick={onClose} className="fixed inset-0 z-30 bg-ink/40 lg:hidden" />}
    <aside className={`fixed inset-y-0 left-0 z-40 flex w-[254px] flex-col bg-ink px-4 pb-4 pt-6 text-white transition-transform lg:static lg:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}>
      <div className="flex items-center justify-between px-3"><BrandMark dark /><button aria-label="Close navigation" onClick={onClose} className="rounded-lg p-1 text-slate-400 hover:bg-white/10 lg:hidden"><X size={18} /></button></div>
      <div className="mt-10 px-3"><p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Care workspace</p></div>
      <nav className="mt-3 space-y-1">{primaryNav.map((item) => <NavItem key={item.to} {...item} />)}</nav>
      <div className="mt-7 border-t border-white/10 pt-6"><div className="px-3"><p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Admin</p></div><NavItem label="Data governance" to="/datasets" icon={Database} /></div>
      <div className="mt-auto rounded-2xl border border-white/10 bg-white/[0.04] p-3"><div className="flex items-center gap-2.5"><div className="h-8 w-8 rounded-full bg-gradient-to-br from-teal-300 to-teal-700 p-[2px]"><div className="flex h-full w-full items-center justify-center rounded-full bg-ink text-[10px] font-bold">CT</div></div><div className="min-w-0"><p className="truncate text-xs font-bold">Signed-in user</p><p className="truncate text-[10px] text-slate-400">Care team</p></div><button aria-label="Sign out" onClick={() => { localStorage.removeItem('retina_nexus_access_token'); navigate('/login'); }} className="ml-auto text-slate-500 hover:text-white"><LogOut size={15} /></button></div></div>
    </aside>
  </>;
}

function NavItem({ label, to, icon: Icon }: { label: string; to: string; icon: typeof Activity }) {
  return <NavLink to={to} className={({ isActive }) => `group flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13px] font-semibold transition ${isActive ? 'bg-teal-500 text-white shadow-[0_8px_18px_rgba(34,170,165,0.18)]' : 'text-slate-400 hover:bg-white/[0.06] hover:text-white'}`}><Icon size={17} strokeWidth={1.9} /><span className="flex-1">{label}</span></NavLink>;
}
