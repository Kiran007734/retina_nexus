const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1';

function authHeaders(): HeadersInit {
  const token = localStorage.getItem('retina_nexus_access_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function requestError(response: Response, fallback: string) {
  const detail = await response.json().catch(() => null);
  return new Error(detail?.detail ?? fallback);
}

export async function login(email: string, password: string) {
  const response = await fetch(`${API_URL}/auth/login`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) });
  if (!response.ok) throw await requestError(response, 'Unable to sign in');
  const result = await response.json() as { access_token: string };
  localStorage.setItem('retina_nexus_access_token', result.access_token);
  return result;
}

export function logout() { localStorage.removeItem('retina_nexus_access_token'); }

export type QualityIssue = { type: string; severity: string; message: string; recommendation: string };
export type QualityResult = {
  image_id: string;
  quality_decision: 'GRADABLE' | 'BORDERLINE' | 'UNGRADABLE';
  quality_score: number;
  final_quality_score: number;
  component_scores: Record<string, number>;
  metrics: Record<string, number>;
  issues: QualityIssue[];
  recommended_action: string;
  enhancement_applied: boolean;
  enhancement_passes: number;
  recheck_score?: number | null;
  recheck_decision?: string | null;
  recheck_issues: QualityIssue[];
  next_action: 'CONTINUE_SCREENING' | 'ENHANCE_AND_REASSESS' | 'RECAPTURE_IMAGE';
  input_metadata: { width: number; height: number; channels: number; format: string; camera_metadata?: Record<string, string> };
  feature_vector: Record<string, number>;
};

export type ClassificationResult = {
  image_id: string;
  screening_session_id: string;
  predicted_grade: number;
  predicted_grade_label: string;
  probabilities: Record<string, number>;
  referable_dr: boolean;
  referable_probability: number;
  raw_confidence: number;
  model_name: string;
  model_version: string;
  backbone: string;
  referable_mapping: { name: string; referable_grades: number[] };
  hierarchical_probabilities: Record<string, Record<string, number>>;
  ordinal_mode: boolean;
  note: string;
};

export type EvidenceModule = {
  module: string;
  category: string;
  status: string;
  supported: boolean;
  implementation: string;
  confidence?: number | null;
  count?: number | null;
  mask_data_uri?: string | null;
  probability_map_data_uri?: string | null;
  overlay_data_uri?: string | null;
  bounding_regions: Array<{ x: number; y: number; width: number; height: number; score?: number; area?: number }>;
  landmarks: Array<Record<string, number | string>>;
  issues: Array<{ type: string; message: string }>;
  metadata: Record<string, unknown>;
};

export type EvidenceAnalysisResult = {
  image_id: string;
  screening_session_id: string;
  status: string;
  image_metadata: { width: number; height: number; working_width: number; working_height: number; channels: number; format: string };
  coarse_to_fine: {
    global_context: Record<string, number>;
    suspicious_region_proposals: EvidenceModule['bounding_regions'];
    high_resolution_patch_extraction: { patch_size: number; patches: Array<Record<string, number>> };
    local_analysis: { modules: string[]; note: string };
  };
  modules: Record<string, EvidenceModule>;
  anatomical_landmarks: Array<Record<string, number | string>>;
  evidence_map_data_uri?: string | null;
  dataset_support: Record<string, Record<string, { status: string; reason: string; annotation_file_count?: number }>>;
  note: string;
};

export type ExplainabilityResult = {
  image_id: string;
  screening_session_id: string;
  predicted_class: number;
  predicted_class_label: string;
  model_version: string;
  classification: {
    predicted_grade: number;
    predicted_grade_label: string;
    probabilities: Record<string, number>;
    referable_dr: boolean;
    referable_probability: number;
    raw_confidence: number;
    model_name: string;
    model_version: string;
    backbone: string;
  };
  grad_cam: {
    heatmap_data_uri: string;
    overlay_data_uri: string;
    normalized_attention_map_data_uri: string;
    target_class: number;
    target_layer: string;
    map_width: number;
    map_height: number;
  };
  lesion_evidence_map_data_uri?: string | null;
  attention_lesion_agreement: {
    status: string;
    score?: number | null;
    metrics: Record<string, number | null>;
    reason: string;
    note: string;
  };
  explanation_stability: {
    status: string;
    reason?: string;
    prediction_stability?: number | null;
    grad_cam_stability?: number | null;
    variants?: Array<Record<string, string | number | boolean>>;
    note?: string;
  };
  counterfactual: {
    status: string;
    experimental: boolean;
    reason?: string;
    selected_region?: string;
    masked_region_data_uri?: string;
    original_predicted_grade?: number;
    counterfactual_predicted_grade?: number;
    predicted_grade_changed?: boolean;
    original_target_probability?: number;
    counterfactual_target_probability?: number;
    target_probability_delta?: number;
    note?: string;
  };
  note: string;
};

export type TrustResult = {
  image_id: string;
  screening_session_id: string;
  trust_score: number;
  trust_category: 'TRUSTED' | 'UNCERTAIN' | 'UNRELIABLE';
  contributing_factors: Array<{ factor: string; score: number; raw_value?: number | null; weight: number; contribution: number; status: string; explanation: string }>;
  risk_flags: Array<{ code: string; severity: string; reason: string }>;
  recommended_action: string;
  calibration: Record<string, unknown>;
  uncertainty: Record<string, unknown>;
  model_disagreement: Record<string, unknown>;
  ood: Record<string, unknown>;
  signal_snapshot: Record<string, unknown>;
  configuration: { version: string; weights: Record<string, number>; [key: string]: unknown };
  reason_summary: string[];
  note: string;
};

export async function getHealth() {
  const response = await fetch(`${API_URL}/health`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'API unavailable');
  return response.json() as Promise<{ status: string; database: string }>;
}

export async function createPatient(payload: { anonymized_identifier: string; age_group?: string }) {
  const response = await fetch(`${API_URL}/patients`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw await requestError(response, 'Unable to create patient');
  return response.json();
}

export async function getPatients() {
  const response = await fetch(`${API_URL}/patients`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load patients');
  return response.json() as Promise<Array<{ id: string; anonymized_identifier: string; age_group?: string | null; created_at: string }>>;
}

export async function getDatasets() {
  const response = await fetch(`${API_URL}/datasets`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load dataset registry');
  return response.json() as Promise<import('../types').DatasetRecord[]>;
}

export async function getDatasetStatistics(datasetId: string) {
  const response = await fetch(`${API_URL}/datasets/${encodeURIComponent(datasetId)}/statistics`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load dataset statistics');
  return response.json() as Promise<import('../types').DatasetStatisticsRecord>;
}

export async function uploadFundusImage(patientId: string, eye: 'left' | 'right', file: File) {
  const body = new FormData();
  body.append('image', file);
  const response = await fetch(`${API_URL}/images/upload?patient_id=${encodeURIComponent(patientId)}&eye=${eye}`, { method: 'POST', headers: authHeaders(), body });
  if (!response.ok) throw await requestError(response, 'Unable to upload image');
  return response.json() as Promise<{ image_id: string }>;
}

export async function assessImageQuality(imageId: string) {
  const response = await fetch(`${API_URL}/images/${imageId}/quality`, { method: 'POST' });
  if (!response.ok) throw await requestError(response, 'Unable to assess image quality');
  return response.json() as Promise<QualityResult>;
}

export async function classifyImage(imageId: string, screeningSessionId?: string) {
  const response = await fetch(`${API_URL}/screening/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId, screening_session_id: screeningSessionId }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? 'Unable to classify image');
  }
  return response.json() as Promise<ClassificationResult>;
}

export async function analyzeStructures(imageId: string, screeningSessionId?: string) {
  const response = await fetch(`${API_URL}/screening/analyze-structures`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId, screening_session_id: screeningSessionId }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? 'Unable to analyze retinal structures');
  }
  return response.json() as Promise<EvidenceAnalysisResult>;
}

export async function explainImage(imageId: string, screeningSessionId?: string, runStability?: boolean, runCounterfactual?: boolean) {
  const response = await fetch(`${API_URL}/screening/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id: imageId, screening_session_id: screeningSessionId, run_stability: runStability, run_counterfactual: runCounterfactual }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? 'Unable to generate explainability output');
  }
  return response.json() as Promise<ExplainabilityResult>;
}

export async function assessTrust(imageId: string, screeningSessionId?: string, modelPredictions?: Array<{ model_version: string; predicted_grade: number; predicted_grade_label?: string; probabilities?: Record<string, number> }>) {
  const response = await fetch(`${API_URL}/screening/trust`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ image_id: imageId, screening_session_id: screeningSessionId, model_predictions: modelPredictions ?? [] }),
  });
  if (!response.ok) throw await requestError(response, 'Unable to calculate RetinaGuard self-check');
  return response.json() as Promise<TrustResult>;
}

export function imageContentUrl(imageId: string, variant: 'original' | 'enhanced' = 'original') {
  return `${API_URL}/images/${imageId}/content?variant=${variant}`;
}

export type ScreeningRun = {
  screening_id: string;
  screening_session_id: string;
  patient_id: string;
  image_id: string;
  status: 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | string;
  stage_status: Record<string, string>;
  stage_metrics: Record<string, { started_at?: string; completed_at?: string; duration_ms?: number | null }>;
  stage_errors: Record<string, unknown>;
  quality?: Record<string, any> | null;
  classification?: ClassificationResult | null;
  lesions?: { status?: string; modules?: Record<string, EvidenceModule>; evidence_map_data_uri?: string | null; note?: string } | null;
  explainability?: ExplainabilityResult | null;
  retinaguard?: TrustResult | null;
  triage?: { recommendation?: string; priority?: string; reasons?: string[]; note?: string; clinical_ai_started?: boolean } | null;
  model_versions: Record<string, any>;
  error?: { stage?: string; type?: string; message?: string } | null;
  message: string;
};

export async function runScreening(imageId: string, options?: { screening_session_id?: string; run_stability?: boolean; run_counterfactual?: boolean }) {
  const response = await fetch(`${API_URL}/screening/run`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders() }, body: JSON.stringify({ image_id: imageId, ...options }) });
  if (!response.ok) throw await requestError(response, 'Unable to run screening pipeline');
  return response.json() as Promise<ScreeningRun>;
}

export async function getScreeningRun(screeningId: string) {
  const response = await fetch(`${API_URL}/screening/${screeningId}`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load screening run');
  return response.json() as Promise<ScreeningRun>;
}

export async function getScreeningHistory() {
  const response = await fetch(`${API_URL}/screening/history`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load screening history');
  return response.json() as Promise<import('../types').ScreeningHistoryItem[]>;
}

export type AnalyticsOverview = { total_screenings: number; today_screenings: number; referable_cases: number; human_review_cases: number; ungradable_images: number; completed_screenings: number; status_distribution: Record<string, number>; severity_distribution: Record<string, number>; recent_activity: Array<Record<string, any>>; system_health: Record<string, string> };

export async function getAnalyticsOverview() {
  const response = await fetch(`${API_URL}/analytics/overview`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load analytics');
  return response.json() as Promise<AnalyticsOverview>;
}

export type Review = { review_id: string; session_id: string; reviewer_id: string; reviewer_name?: string | null; decision: string; modified_grade?: number | null; comments?: string | null; created_at?: string | null };
export type ReviewQueueItem = { session_id: string; patient_id: string; image_id: string; eye: string; status: string; trust_category?: string | null; trust_score?: number | null; predicted_grade?: number | null; predicted_grade_label?: string | null; referable_dr?: boolean | null; reason: string; created_at?: string | null; review?: Review | null };

export async function getReviewQueue() {
  const response = await fetch(`${API_URL}/reviews/queue`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load clinical review queue');
  return response.json() as Promise<ReviewQueueItem[]>;
}

export async function getSessionReviews(sessionId: string) {
  const response = await fetch(`${API_URL}/reviews/${sessionId}`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load clinical reviews');
  return response.json() as Promise<Review[]>;
}

export async function submitReview(sessionId: string, payload: { decision: 'approve' | 'modify' | 'reject' | 'request_recapture'; modified_grade?: number; comments?: string }) {
  const response = await fetch(`${API_URL}/reviews/${sessionId}`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders() }, body: JSON.stringify(payload) });
  if (!response.ok) throw await requestError(response, 'Unable to save clinician review');
  return response.json() as Promise<Review>;
}

export type Report = { report_id: string; session_id: string; status: string; download_url?: string | null; created_at?: string | null; report?: Record<string, any> | null };

export async function getReports() {
  const response = await fetch(`${API_URL}/reports`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load reports');
  return response.json() as Promise<Report[]>;
}

export async function generateReport(sessionId: string) {
  const response = await fetch(`${API_URL}/reports/generate`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...authHeaders() }, body: JSON.stringify({ session_id: sessionId }) });
  if (!response.ok) throw await requestError(response, 'Unable to generate report');
  return response.json() as Promise<Report>;
}

export function reportPdfUrl(reportId: string) { return `${API_URL}/reports/${reportId}/pdf`; }

export async function getModels() {
  const response = await fetch(`${API_URL}/models`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load model registry');
  return response.json() as Promise<Array<Record<string, any>>>;
}

export async function getMonitoringSummary(days = 30) {
  const response = await fetch(`${API_URL}/monitoring/summary?days=${days}`, { headers: authHeaders() });
  if (!response.ok) throw await requestError(response, 'Unable to load monitoring summary');
  return response.json() as Promise<Record<string, any>>;
}

export type DemoScenario = {
  scenario_id: string;
  image_label: string;
  title: string;
  summary: string;
  expected_category?: string;
  expected_action?: string;
  quality?: Record<string, any> | null;
  classification?: Record<string, any> | null;
  lesions?: Record<string, any> | null;
  explainability?: Record<string, any> | null;
  retinaguard?: Record<string, any> | null;
  triage?: Record<string, any> | null;
  model_versions?: Record<string, any>;
};

export async function getDemoScenarios() {
  const response = await fetch(`${API_URL}/demo/scenarios`);
  if (!response.ok) throw await requestError(response, 'Demo mode is disabled or unavailable');
  return response.json() as Promise<{ demo_mode: boolean; sample_data: boolean; scenarios: DemoScenario[]; note: string }>;
}

export async function runDemoScenario(scenarioId: string) {
  const response = await fetch(`${API_URL}/demo/scenarios/${encodeURIComponent(scenarioId)}/run`, { method: 'POST' });
  if (!response.ok) throw await requestError(response, 'Unable to run demo scenario');
  return response.json() as Promise<{ demo_mode: boolean; sample_data: boolean; persisted_to_clinical_records: boolean; demo_run_id: string; scenario: DemoScenario; note: string }>;
}
