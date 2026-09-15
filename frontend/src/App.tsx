import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { RequireAuth } from './components/RequireAuth';
import { AppLayout } from './layouts/AppLayout';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { DashboardPage } from './pages/DashboardPage';
import { DatasetManagementPage } from './pages/DatasetManagementPage';
import { DemoPage } from './pages/DemoPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { ForgotPasswordPage } from './pages/ForgotPasswordPage';
import { LoginPage } from './pages/LoginPage';
import { LandingPage } from './pages/LandingPage';
import { MonitoringPage } from './pages/MonitoringPage';
import { NewScreeningPage } from './pages/NewScreeningPage';
import { PatientsPage } from './pages/PatientsPage';
import { ReportsPage } from './pages/ReportsPage';
import { ResultsPage } from './pages/ResultsPage';
import { ReviewPage } from './pages/ReviewPage';
import { ScreeningHistoryPage } from './pages/ScreeningHistoryPage';
import { SettingsPage } from './pages/SettingsPage';
import { SignupPage } from './pages/SignupPage';

export default function App() {
  return <Routes>
    <Route path="/" element={<LandingPage />} />
    <Route path="/login" element={<LoginPage />} />
    <Route path="/signup" element={<SignupPage />} />
    <Route path="/forgot-password" element={<ForgotPasswordPage />} />
    <Route element={<RequireAuth />}>
      <Route path="/platform" element={<AppLayout />}>
        <Route index element={<DashboardPage />} />
        <Route path="screening/new" element={<NewScreeningPage />} />
        <Route path="screening/results" element={<ResultsPage />} />
        <Route path="screening/explain" element={<ExplainabilityPage />} />
        <Route path="history" element={<ScreeningHistoryPage />} />
        <Route path="patients" element={<PatientsPage />} />
        <Route path="review" element={<ReviewPage />} />
        <Route path="reports" element={<ReportsPage />} />
        <Route path="analytics" element={<AnalyticsPage />} />
        <Route path="monitoring" element={<MonitoringPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="datasets" element={<DatasetManagementPage />} />
        <Route path="demo" element={<DemoPage />} />
      </Route>
      <Route path="/dashboard" element={<Navigate to="/platform" replace />} />
      <Route path="/screening/*" element={<LegacyPlatformRedirect />} />
      {['history', 'patients', 'review', 'reports', 'analytics', 'monitoring', 'settings', 'datasets', 'demo'].map((path) => (
        <Route key={path} path={`/${path}`} element={<LegacyPlatformRedirect />} />
      ))}
    </Route>
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes>;
}

function LegacyPlatformRedirect() {
  const location = useLocation();
  return <Navigate to={`/platform${location.pathname}${location.search}${location.hash}`} replace state={location.state} />;
}
