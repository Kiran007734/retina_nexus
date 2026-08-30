export type StatusTone = 'success' | 'warning' | 'danger' | 'neutral' | 'teal';

export type Patient = {
  id: string;
  anonymized_identifier: string;
  age_group?: string | null;
  created_at: string;
};

export type ScreeningRecord = {
  id: string;
  patientId: string;
  patientLabel: string;
  eye: 'OD' | 'OS';
  time: string;
  status: 'Review needed' | 'Ready' | 'Processing';
  severity?: string;
};

export type ScreeningHistoryItem = {
  screening_id: string;
  patient_id: string;
  image_id: string;
  eye: string;
  status: string;
  trust_category?: string | null;
  trust_score?: number | null;
  predicted_grade?: number | null;
  predicted_grade_label?: string | null;
  referable_dr?: boolean | null;
  triage_recommendation?: string | null;
  created_at?: string | null;
};

export type ModelHealth = {
  label: string;
  version: string;
  status: 'Active' | 'Standby' | 'Not configured';
  value: string;
  note: string;
};

export type DatasetRecord = {
  id?: string;
  slug: string;
  name: string;
  purpose: string;
  status: 'not_acquired' | 'available' | 'validating' | 'ready' | 'blocked';
  availability_status?: 'AVAILABLE' | 'PARTIALLY AVAILABLE' | 'MISSING' | 'INVALID';
  raw_path: string;
  latest_version?: string | null;
  image_count?: number | null;
  readiness_score?: number | null;
};

export type DatasetStatisticsRecord = {
  dataset_id: string;
  dataset_version?: string | null;
  total_files: number;
  readable_files: number;
  corrupted_files: number;
  duplicate_exact_count: number;
  duplicate_perceptual_count: number;
  class_distribution?: Record<string, number> | null;
  readiness_score?: number | null;
};
