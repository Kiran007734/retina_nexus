import { useEffect, useRef, useState } from 'react';
import { CheckCircle2, FileImage, UploadCloud, X } from 'lucide-react';

export function UploadArea({ onFile }: { onFile: (file: File | null) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  function accept(next: File | null) {
    if (!next) return;
    if (!['image/jpeg', 'image/png'].includes(next.type)) return;
    setPreviewUrl((previous) => { if (previous) URL.revokeObjectURL(previous); return URL.createObjectURL(next); });
    setFile(next); onFile(next);
  }

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  return <div onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); accept(event.dataTransfer.files[0]); }} className={`rounded-2xl border-2 border-dashed p-8 text-center transition ${dragging ? 'border-teal-500 bg-teal-50' : file ? 'border-teal-300 bg-teal-50/40' : 'border-slate-200 bg-mist hover:border-teal-300 hover:bg-teal-50/40'}`}>
    {file ? <div className="flex flex-col items-center"><div className="h-28 w-44 overflow-hidden rounded-xl bg-white shadow-sm"><img src={previewUrl ?? undefined} alt="Selected fundus preview" className="h-full w-full object-cover" /></div><div className="mt-4 flex items-center gap-2 text-sm font-bold text-ink"><CheckCircle2 size={16} className="text-teal-600" />{file.name}</div><p className="mt-1 text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB · ready for quality check</p><button type="button" onClick={() => { setPreviewUrl(null); setFile(null); onFile(null); }} className="mt-4 inline-flex items-center gap-1 text-xs font-bold text-slate-500 hover:text-rose-500"><X size={13} /> Remove file</button></div> : <><div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-white text-teal-600 shadow-sm"><UploadCloud size={27} /></div><p className="mt-4 text-sm font-bold text-ink">Drop a fundus image here</p><p className="mt-1 text-xs text-slate-400">JPEG or PNG · up to 15 MB</p><button type="button" onClick={() => input.current?.click()} className="btn-secondary mt-5 !px-4 !py-2 text-xs">Browse files</button><input ref={input} type="file" accept="image/jpeg,image/png" className="hidden" onChange={(event) => accept(event.target.files?.[0] ?? null)} /></>}
  </div>;
}
