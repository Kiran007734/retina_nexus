import { useEffect, useState } from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { getAccessToken, getCurrentUser, logout } from '../services/api';

export function RequireAuth() {
  const location = useLocation();
  const [state, setState] = useState<'checking' | 'authenticated' | 'anonymous'>(getAccessToken() ? 'checking' : 'anonymous');

  useEffect(() => {
    let active = true;
    const expire = () => { if (active) setState('anonymous'); };
    window.addEventListener('retina-nexus:auth-expired', expire);

    if (!getAccessToken()) {
      setState('anonymous');
    } else {
      getCurrentUser()
        .then(() => { if (active) setState('authenticated'); })
        .catch(() => {
          logout();
          if (active) setState('anonymous');
        });
    }

    return () => {
      active = false;
      window.removeEventListener('retina-nexus:auth-expired', expire);
    };
  }, []);

  if (state === 'checking') {
    return <div className="flex min-h-screen items-center justify-center bg-mist"><div className="text-center"><span className="mx-auto block h-8 w-8 animate-spin rounded-full border-2 border-teal-100 border-t-teal-600" /><p className="mt-4 text-xs font-bold text-slate-500">Verifying secure session</p></div></div>;
  }

  if (state === 'anonymous') {
    return <Navigate to="/login" replace state={{ from: `${location.pathname}${location.search}${location.hash}` }} />;
  }

  return <Outlet />;
}
