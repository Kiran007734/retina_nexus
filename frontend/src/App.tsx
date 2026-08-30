import { Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from './layouts/AppLayout';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { DashboardPage } from './pages/DashboardPage';
import { DatasetManagementPage } from './pages/DatasetManagementPage';
import { DemoPage } from './pages/DemoPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { LoginPage } from './pages/LoginPage';
import { MonitoringPage } from './pages/MonitoringPage';
import { NewScreeningPage } from './pages/NewScreeningPage';
import { PatientsPage } from './pages/PatientsPage';
import { ReportsPage } from './pages/ReportsPage';
import { ResultsPage } from './pages/ResultsPage';
import { ReviewPage } from './pages/ReviewPage';
import { ScreeningHistoryPage } from './pages/ScreeningHistoryPage';
import { SettingsPage } from './pages/SettingsPage';

export default function App() {
  return <Routes>
    <Route path="/login" element={<LoginPage />} />
    <Route element={<AppLayout />}>
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
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes>;
}
