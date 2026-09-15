import { ArrowLeft, ArrowRight, CheckCircle2, Mail } from 'lucide-react';
import { FormEvent, useState } from 'react';
import { Link } from 'react-router-dom';
import { AuthLayout, FieldError, authInputClass } from '../components/AuthLayout';
import { requestPasswordReset } from '../services/api';

export function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!/^\S+@\S+\.\S+$/.test(email)) { setError('Enter a valid work email.'); return; }
    setError(''); setLoading(true);
    try { await requestPasswordReset(email.trim().toLowerCase()); setSubmitted(true); }
    catch (requestError) { setError(requestError instanceof Error ? requestError.message : 'Unable to request password reset.'); }
    finally { setLoading(false); }
  }

  return <AuthLayout eyebrow="Secure workspace" title={submitted ? 'Check your inbox' : 'Reset your password'} description={submitted ? 'Password reset instructions have been sent if an account exists for this email.' : "Enter your work email and we'll help you restore access."}>
    {submitted ? <div className="mt-8"><div className="flex items-start gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-5 text-emerald-900"><CheckCircle2 size={21} className="mt-0.5 shrink-0" /><div><p className="text-sm font-extrabold">Request received</p><p className="mt-1 text-xs leading-5">For privacy, RETINA NEXUS uses the same confirmation whether or not an account exists.</p></div></div><Link to="/login" className="btn-secondary mt-6 w-full"><ArrowLeft size={16} /> Back to sign in</Link></div> : <form onSubmit={submit} noValidate className="mt-8 space-y-5"><label className="block"><span className="mb-2 block text-xs font-bold text-ink">Work email</span><div className="relative"><input value={email} onChange={(event) => setEmail(event.target.value)} type="email" autoComplete="email" className={`${authInputClass} pl-11`} placeholder="you@clinic.org" aria-invalid={Boolean(error)} /><Mail size={16} className="absolute left-4 top-3.5 text-slate-300" /></div><FieldError message={error} /></label><button disabled={loading} className="btn-primary w-full py-3 disabled:cursor-not-allowed disabled:opacity-60">{loading ? 'Sending request…' : 'Send reset link'} <ArrowRight size={16} /></button><Link to="/login" className="btn-quiet w-full"><ArrowLeft size={15} /> Back to sign in</Link></form>}
  </AuthLayout>;
}
