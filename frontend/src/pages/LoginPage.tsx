import { ArrowRight, Eye, EyeOff, LockKeyhole, ShieldCheck } from 'lucide-react';
import { FormEvent, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { AuthLayout, FieldError, authInputClass } from '../components/AuthLayout';
import { login } from '../services/api';

type LoginState = { from?: string; accountCreated?: boolean; email?: string } | null;

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const routeState = location.state as LoginState;
  const [email, setEmail] = useState(routeState?.email ?? '');
  const [password, setPassword] = useState('');
  const [remember, setRemember] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState<{ email?: string; password?: string }>({});
  const [requestError, setRequestError] = useState('');
  const [loading, setLoading] = useState(false);

  function validate() {
    const next: { email?: string; password?: string } = {};
    if (!/^\S+@\S+\.\S+$/.test(email)) next.email = 'Enter a valid work email.';
    if (password.length < 8) next.password = 'Password must contain at least 8 characters.';
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setRequestError('');
    if (!validate()) return;
    setLoading(true);
    try {
      await login(email.trim().toLowerCase(), password, remember);
      navigate(routeState?.from ?? '/platform', { replace: true });
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : 'Unable to sign in.');
    } finally {
      setLoading(false);
    }
  }

  return <AuthLayout eyebrow="Secure workspace" title="Sign in to continue" description="Access screening, evidence verification, and clinician review tools.">
    {routeState?.accountCreated && <div role="status" className="mt-6 rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-xs font-semibold leading-5 text-emerald-800">Workspace account created. Sign in with your new credentials.</div>}
    <form onSubmit={submit} noValidate className="mt-8 space-y-5">
      <label className="block"><span className="mb-2 block text-xs font-bold text-ink">Work email</span><input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} className={authInputClass} placeholder="you@clinic.org" aria-invalid={Boolean(errors.email)} /><FieldError message={errors.email} /></label>
      <label className="block"><span className="mb-2 block text-xs font-bold text-ink">Password</span><div className="relative"><input type={showPassword ? 'text' : 'password'} autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} className={`${authInputClass} px-11`} placeholder="Enter your password" aria-invalid={Boolean(errors.password)} /><LockKeyhole size={16} className="absolute left-4 top-3.5 text-slate-300" /><button type="button" onClick={() => setShowPassword((value) => !value)} className="absolute right-3 top-2.5 rounded-lg p-1.5 text-slate-400 hover:bg-mist hover:text-ink" aria-label={showPassword ? 'Hide password' : 'Show password'}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button></div><FieldError message={errors.password} /></label>
      {requestError && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs leading-5 text-rose-800">{requestError}</div>}
      <div className="flex items-center justify-between gap-3 text-xs"><label className="flex items-center gap-2 text-slate-500"><input type="checkbox" checked={remember} onChange={(event) => setRemember(event.target.checked)} className="rounded border-slate-300 text-teal-600 focus:ring-teal-200" /> Remember me</label><Link to="/forgot-password" className="font-bold text-teal-700 hover:text-teal-600">Forgot password?</Link></div>
      <button disabled={loading} className="btn-primary w-full py-3 disabled:cursor-not-allowed disabled:opacity-60">{loading ? 'Signing in…' : 'Sign in to workspace'} <ArrowRight size={16} /></button>
    </form>
    <p className="mt-5 text-center text-xs text-slate-500">Don't have an account? <Link to="/signup" className="font-bold text-teal-700 hover:text-teal-600">Create account</Link></p>
    <div className="mt-8 flex items-start gap-3 rounded-xl border border-teal-100 bg-teal-50 p-3 text-xs leading-5 text-teal-800"><ShieldCheck size={16} className="mt-0.5 shrink-0 text-teal-600" /><span>Access is protected by workspace roles and audit logging. This prototype does not claim regulatory certification.</span></div>
  </AuthLayout>;
}
