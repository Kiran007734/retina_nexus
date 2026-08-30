import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from '../components/Header';
import { Sidebar } from '../components/Sidebar';

export function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  return <div className="flex min-h-screen bg-mist"><Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} /><div className="flex min-w-0 flex-1 flex-col"><Header onMenu={() => setSidebarOpen(true)} /><main className="flex-1 px-4 py-6 sm:px-8 sm:py-8 lg:px-10 lg:py-10"><div className="mx-auto max-w-[1480px]"><Outlet /></div></main><footer className="px-5 pb-6 text-center text-[10px] text-slate-400 sm:px-8 lg:px-10">RETINA-NEXUS · Care intelligence workspace · Prototype · AI output requires clinical review</footer></div></div>;
}
