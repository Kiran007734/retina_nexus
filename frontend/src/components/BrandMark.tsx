import { Eye } from 'lucide-react';

export function BrandMark({ dark = false }: { dark?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500 text-white shadow-[0_8px_20px_rgba(34,170,165,0.24)]">
        <Eye size={21} strokeWidth={2.4} />
      </div>
      <div>
        <p className={`section-title text-[15px] font-extrabold tracking-[0.16em] ${dark ? 'text-white' : 'text-ink'}`}>RETINA</p>
        <p className={`-mt-1 text-[10px] font-bold tracking-[0.34em] ${dark ? 'text-teal-100' : 'text-teal-600'}`}>NEXUS</p>
      </div>
    </div>
  );
}
