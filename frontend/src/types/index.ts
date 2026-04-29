// OncoIDT TypeScript interfaces — mirrors shared/models.py and shared/schemas.py

// ---------------------------------------------------------------------------
// Supporting leaf types
// ---------------------------------------------------------------------------

export interface RuleOverride {
  rule_id: string
  threshold_value: number
  triggered_value: number
  score_floor: number
}

export interface DataQualityFlag {
  flag_type: string
  field_name: string
  message: string
  severity: 'info' | 'warning' | 'error'
}

export interface VitalsSnapshot {
  temperature_c: number | null
  heart_rate_bpm: number | null
  respiratory_rate_rpm: number | null
  sbp_mmhg: number | null
  dbp_mmhg: number | null
  spo2_pct: number | null
  gcs: number | null
  timestamp: string // ISO 8601
}

export interface LabSnapshot {
  anc: number | null // × 10⁹/L
  wbc: number | null
  lymphocytes: number | null
  crp_mg_l: number | null
  procalcitonin_ug_l: number | null
  lactate_mmol_l: number | null
  creatinine_umol_l: number | null
  timestamp: string
}

export interface RiskScore {
  score: number // [0, 1]
  uncertainty_lower: number
  uncertainty_upper: number
  forecast_horizon_hours: number
  model_version: string
  feature_snapshot_id: string
  rule_overrides: RuleOverride[]
  timestamp: string
}

export interface SurvivalEstimate {
  median_hours: number
  ci_80_lower_hours: number
  ci_80_upper_hours: number
  event_type: 'infection' | 'deterioration' | 'icu_transfer'
  model_version: string
  timestamp: string
}

export interface BedState {
  bed_id: string
  room_id: string
  zone: string
  patient_id: string | null
  infection_risk_score: number | null
  deterioration_risk_score: number | null
  exposure_flag: boolean
  last_score_timestamp: string | null
}

export interface ExposureEvent {
  event_id: string
  source_patient_id: string
  pathogen: string | null
  affected_patient_ids: string[]
  timestamp: string
  confidence: number
}

export interface EnvironmentalContext {
  air_quality_index: number | null
  temperature_celsius: number | null
  humidity_pct: number | null
  timestamp: string
}

// ---------------------------------------------------------------------------
// Canonical record payload types
// ---------------------------------------------------------------------------

export interface LabRecord {
  loinc_code: string
  source_code: string
  source_system_code: string
  value_numeric: number | null
  value_text: string | null
  unit: string
  reference_range_low: number | null
  reference_range_high: number | null
  abnormal_flag: string | null
}

export interface VitalRecord {
  loinc_code: string
  value_numeric: number | null
  value_text: string | null
  unit: string
}

export interface MedicationRecord {
  rxnorm_code: string
  source_code: string
  drug_name: string
  dose_mg: number | null
  route: string
  is_chemotherapy: boolean
  chemo_regimen_code: string | null
  administration_timestamp: string
}

export interface ObservationRecord {
  observation_type: string
  value_text: string | null
  value_numeric: number | null
  unit: string | null
}

export interface ClinicalEventRecord {
  event_type: string
  description: string | null
  metadata: Record<string, unknown>
}

export interface NoteMetadataRecord {
  note_type: string
  author_role: string
  word_count: number | null
}

export type RecordPayload =
  | VitalRecord
  | LabRecord
  | MedicationRecord
  | ObservationRecord
  | ClinicalEventRecord
  | NoteMetadataRecord

export interface CanonicalRecord {
  record_id: string
  patient_id: string
  source_system: string
  source_record_id: string
  record_type: 'vital' | 'lab' | 'medication' | 'observation' | 'event' | 'note_metadata'
  timestamp_utc: string
  ingested_at: string
  data_quality_flags: DataQualityFlag[]
  payload: RecordPayload
}

// ---------------------------------------------------------------------------
// Top-level twin types
// ---------------------------------------------------------------------------

export type ChemoCyclePhase = 'pre' | 'nadir' | 'recovery' | 'off'
export type PatientStatus = 'active' | 'archived' | 'transferred'
export type PatientSex = 'M' | 'F' | 'O' | 'U'

export interface PatientTwin {
  patient_id: string
  mrn: string
  ward_id: string
  bed_id: string | null
  admission_timestamp: string
  discharge_timestamp: string | null
  status: PatientStatus

  age_years: number
  sex: PatientSex
  primary_diagnosis_icd10: string
  comorbidities: string[]

  chemo_regimen: string
  chemo_cycle_number: number
  chemo_cycle_phase: ChemoCyclePhase
  days_since_last_chemo_dose: number
  cumulative_dose_mg_m2: number
  immunosuppression_score: number

  vitals: VitalsSnapshot
  labs: LabSnapshot

  infection_risk_scores: Record<string, RiskScore>
  deterioration_risk_scores: Record<string, RiskScore>
  survival_estimate: SurvivalEstimate | null

  last_updated: string
  data_quality_flags: DataQualityFlag[]
  feature_version: string
}

export interface WardTwin {
  ward_id: string
  ward_name: string
  total_beds: number
  occupied_beds: number
  last_updated: string

  beds: Record<string, BedState>

  ward_infection_risk: number
  ward_deterioration_risk: number
  high_risk_patient_count: number

  environmental: EnvironmentalContext | null

  active_exposure_events: ExposureEvent[]
  recent_confirmed_infections: string[]
}

// ---------------------------------------------------------------------------
// Alert types
// ---------------------------------------------------------------------------

export type AlertPriority = 'Critical' | 'High' | 'Medium' | 'Low'
export type AlertStatus = 'active' | 'acknowledged' | 'snoozed' | 'resolved'

export interface Alert {
  alert_id: string
  patient_id: string
  alert_type: string
  priority: AlertPriority
  status: AlertStatus
  message: string
  score_value: number | null
  score_delta: number | null
  escalation_count: number
  created_at: string
  updated_at: string
  recipient_roles: string[]
  top_features: string[]
}

// ---------------------------------------------------------------------------
// Explanation types
// ---------------------------------------------------------------------------

export interface FeatureAttribution {
  feature_name: string
  shap_value: number
  feature_value: number | string | null
  direction: 'positive' | 'negative'
  rank: number
  nl_sentence: string
}

export interface Explanation {
  explanation_id: string
  patient_id: string
  score_type: 'infection' | 'deterioration'
  forecast_horizon_hours: number
  top_features: FeatureAttribution[]
  model_version: string
  timestamp: string
  is_rule_triggered: boolean
  rule_ids: string[]
}

export interface WardExplanation {
  ward_id: string
  computed_at: string
  top_features: FeatureAttribution[]
}

// ---------------------------------------------------------------------------
// Simulation types
// ---------------------------------------------------------------------------

export type InterventionType = 'antibiotic_administration' | 'dose_modification' | 'isolation_measure'

export interface Intervention {
  type: InterventionType
  parameter: string
  value: string | number | boolean
  apply_at_hours: number
}

export interface SimulationRequest {
  patient_id: string
  interventions: Intervention[]
  horizons: number[]
}

export interface SimulationResult {
  session_id: string
  patient_id: string
  status: 'pending' | 'running' | 'complete' | 'failed'
  baseline_scores: Record<string, RiskScore>
  counterfactual_scores: Record<string, RiskScore>
  delta_explanation: Explanation | null
  created_at: string
  completed_at: string | null
}

// ---------------------------------------------------------------------------
// WebSocket message types
// ---------------------------------------------------------------------------

export type WsMessageType = 'score_update' | 'alert_generated' | 'ward_state_change'

export interface WsScoreUpdate {
  type: 'score_update'
  patient_id: string
  infection_risk_scores: Record<string, RiskScore>
  deterioration_risk_scores: Record<string, RiskScore>
  survival_estimate: SurvivalEstimate | null
  timestamp: string
}

export interface WsAlertGenerated {
  type: 'alert_generated'
  alert: Alert
}

export interface WsWardStateChange {
  type: 'ward_state_change'
  ward: WardTwin
}

export type WsMessage = WsScoreUpdate | WsAlertGenerated | WsWardStateChange

// ---------------------------------------------------------------------------
// API response envelopes
// ---------------------------------------------------------------------------

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}

export interface ErrorResponse {
  detail: string
  code: string | null
}
