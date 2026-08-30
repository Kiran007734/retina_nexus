import { Focus, ImageOff, Maximize2, RotateCcw } from 'lucide-react';

export function ImageViewer({ src, alt = 'Fundus image', compact = false }: { src?: string | null; alt?: string; compact?: boolean }) {
  return <div className={`relative flex items-center justify-center overflow-hidden rounded-2xl bg-[#132643] ${compact ? 'h-44' : 'min-h-[390px]'}`}>
    {src ? <img src={src} alt={alt} className="relative max-h-full max-w-full object-contain" /> : <div className="relative flex flex-col items-center px-8 text-center text-white/75"><ImageOff size={28} /><p className="mt-3 text-sm font-bold text-white">No fundus image loaded</p><p className="mt-1 text-xs">Upload or select an image to begin review.</p></div>}
    <div className="absolute left-4 top-4 rounded-lg bg-black/30 px-2.5 py-1.5 text-[10px] font-bold uppercase tracking-widest text-white/80">{src ? 'Source image' : 'Image unavailable'}</div>
    {src && <div className="absolute bottom-4 right-4 flex gap-1.5"><button aria-label="Focus image" className="rounded-lg bg-white/10 p-2 text-white backdrop-blur hover:bg-white/20"><Focus size={14} /></button><button aria-label="Rotate image" className="rounded-lg bg-white/10 p-2 text-white backdrop-blur hover:bg-white/20"><RotateCcw size={14} /></button><button aria-label="Expand image" className="rounded-lg bg-white/10 p-2 text-white backdrop-blur hover:bg-white/20"><Maximize2 size={14} /></button></div>}
  </div>;
}
