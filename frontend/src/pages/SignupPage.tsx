import { ArrowRight, Eye, EyeOff, ShieldCheck } from 'lucide-react';
import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthLayout, FieldError, authInputClass } from '../components/AuthLayout';
import { ProfessionalRole, signup } from '../services/api';

const roleOptions: { value: ProfessionalRole; label: string }[] = [
  { value: 'ophthalmologist', label: 'Ophthalmologist' },
  { value: 'optometrist', label: 'Optometrist' },
  { value: 'clinician', label: 'Clinician' },
  { value: 'screening_operator', label: 'Screening Operator' },
  { value: 'researcher', label: 'Researcher' },
  { value: 'administrator', label: 'Administrator' },
];

type SignupErrors = Partial<Record<'fullName' | 'email' | 'organization' | 'role' | 'password' | 'confirmation' | 'terms', string>>;

export function SignupPage() {
  const navigate = useNavigate();
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [organization, setOrganization] = useState('');
  const [role, setRole] = useState<ProfessionalRole | ''>('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [terms, setTerms] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState<SignupErrors>({});
  const [requestError, setRequestError] = useState('');
  const [loading, setLoading] = useState(false);

  function validate() {
    const next: SignupErrors = {};
    if (fullName.trim().length < 2) next.fullName = 'Enter your full name.';
    if (!/^\S+@\S+\.\S+$/.test(email)) next.email = 'Enter a valid work email.';
    if (organization.trim().length < 2) next.organization = 'Enter your organization or clinic.';
    if (!role) next.role = 'Select your professional role.';
    if (password.length < 12 || !/[a-z]/.test(password) || !/[A-Z]/.test(password) || !/\d/.test(password) || !/[^A-Za-z0-9]/.test(password)) next.password = 'Use 12+ characters with uppercase, lowercase, number, and symbol.';
    if (confirmation !== password) next.confirmation = 'Passwords do not match.';
    if (!terms) next.terms = 'Accept the prototype terms and privacy notice to continue.';
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setRequestError('');
    if (!validate() || !role) return;
    setLoading(true);
    try {
      await signup({ full_name: fullName.trim(), email: email.trim().toLowerCase(), organization: organization.trim(), professional_role: role, password, terms_accepted: terms });
      navigate('/login', { replace: true, state: { accountCreated: true, email: email.trim().toLowerCase() } });
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : 'Unable to create workspace account.');
    } finally {
      setLoading(false);
    }
  }

  return <AuthLayout eyebrow="Create workspace" title="Create your workspace" description="Set up secure access to retinal screening and clinical review tools.">
    <form onSubmit={submit} noValidate className="mt-8 space-y-4">
      <AuthField label="Full name" value={fullName} onChange={setFullName} autoComplete="name" error={errors.fullName} />
      <AuthField label="Work email" value={email} onChange={setEmail} type="email" autoComplete="email" error={errors.email} />
      <AuthField label="Organization / Clinic" value={organization} onChange={setOrganization} autoComplete="organization" error={errors.organization} />
      <label className="block"><span className="mb-2 block text-xs font-bold text-ink">Role</span><select value={role} onChange={(event) => setRole(event.target.value as ProfessionalRole | '')} className={authInputClass}><option value="">Select a role</option>{roleOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select><FieldError message={errors.role} /></label>
      <label className="block"><span className="mb-2 block text-xs font-bold text-ink">Password</span><div className="relative"><input value={password} onChange={(event) => setPassword(event.target.value)} type={showPassword ? 'text' : 'password'} autoComplete="new-password" maxLength={72} className={`${authInputClass} pr-12`} aria-invalid={Boolean(errors.password)} /><button type="button" onClick={() => setShowPassword((value) => !value)} className="absolute right-3 top-2.5 rounded-lg p-1.5 text-slate-400 hover:bg-mist hover:text-ink" aria-label={showPassword ? 'Hide password' : 'Show password'}>{showPassword ? <EyeOff size={17} /> : <Eye size={17} />}</button></div><FieldError message={errors.password} /></label>
      <AuthField label="Confirm password" value={confirmation} onChange={setConfirmation} type={showPassword ? 'text' : 'password'} autoComplete="new-password" error={errors.confirmation} />
      <label className="flex items-start gap-2.5 text-xs leading-5 text-slate-500"><input type="checkbox" checked={terms} onChange={(event) => setTerms(event.target.checked)} className="mt-1 rounded border-slate-300 text-teal-600 focus:ring-teal-200" /><span>I agree to the prototype terms and privacy notice.</span></label><FieldError message={errors.terms} />
      {requestError && <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs leading-5 text-rose-800">{requestError}</div>}
      <button disabled={loading} className="btn-primary w-full py-3 disabled:cursor-not-allowed disabled:opacity-60">{loading ? 'Creating workspace…' : 'Create workspace'} <ArrowRight size={16} /></button>
    </form>
    <p className="mt-5 text-center text-xs text-slate-500">Already have an account? <Link to="/login" className="font-bold text-teal-700 hover:text-teal-600">Sign in</Link></p>
    <div className="mt-7 flex items-start gap-3 rounded-xl border border-teal-100 bg-teal-50 p-3 text-xs leading-5 text-teal-800"><ShieldCheck size={16} className="mt-0.5 shrink-0 text-teal-600" /><span>Public registration uses least-privilege workspace access. Administrator authorization is never self-granted.</span></div>
  </AuthLayout>;
}

function AuthField({ label, value, onChange, type = 'text', autoComplete, error }: { label: string; value: string; onChange: (value: string) => void; type?: string; autoComplete?: string; error?: string }) {
  return <label className="block"><span className="mb-2 block text-xs font-bold text-ink">{label}</span><input value={value} onChange={(event) => onChange(event.target.value)} type={type} autoComplete={autoComplete} maxLength={type === 'password' ? 72 : undefined} className={authInputClass} aria-invalid={Boolean(error)} /><FieldError message={error} /></label>;
}
