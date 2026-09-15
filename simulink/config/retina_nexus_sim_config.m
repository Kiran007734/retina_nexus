function cfg = retina_nexus_sim_config(overrides)
%RETINA_NEXUS_SIM_CONFIG Central configuration for the system-level twin.
% This file contains operational simulation parameters only. It does not
% change the clinical pipeline, model weights, thresholds, or fusion.

if nargin < 1 || isempty(overrides)
    overrides = struct();
end

cfg = struct();
cfg.schema_version = 'retina-nexus-simulink-config-v1';
cfg.scenario_name = 'BASELINE_RURAL';
cfg.random_seed = 20260914;
cfg.simulation_days = 14;
cfg.simulation_seconds = cfg.simulation_days * 24 * 3600;
cfg.arrival_process = 'POISSON';

% SIH problem-statement workload and derived rates.
cfg.annual_patients = 100000;
cfg.days_per_year = 365;
cfg.peak_multiplier = 1.0;
cfg.required_rate_per_hour = cfg.annual_patients / (cfg.days_per_year * 24);
cfg.arrival_rate_per_hour = cfg.required_rate_per_hour * cfg.peak_multiplier;
cfg.arrival_rate_per_second = cfg.arrival_rate_per_hour / 3600;
cfg.interarrival_seconds = 1 / cfg.arrival_rate_per_second;

% Simulation assumptions, not measured clinical/site data.
cfg.image_size_mb = 5;
cfg.bandwidth_mbps = 10;
cfg.network_latency_ms = 150;
cfg.transmission_variability = 0;
cfg.quality_rejection_rate = 0.08;
cfg.recapture_probability = 0.70;
cfg.max_recaptures = 2;
cfg.review_probability = 0.20;
cfg.referral_probability = 0.25;
cfg.review_time_minutes = 2;

% Measured project runtime values from the pre-Simulink audit.
cfg.timing_mode = 'MEAN';
cfg.measured_runtime_source = 'ml/evaluation/final_validation/runtime_validation.json';
cfg.services = struct();
cfg.services.classification = struct('mean_s',0.209,'worst_s',0.275);
cfg.services.lesion = struct('mean_s',6.28,'worst_s',10.62);
cfg.services.localization = struct('mean_s',0.0595,'worst_s',0.099);
cfg.services.vessel = struct('mean_s',144.5,'worst_s',246.4);
cfg.services.xai_agreement = struct('mean_s',10.2,'worst_s',15.7);
cfg.services.retina_guard = struct('mean_s',0.0088,'worst_s',0.0171);
cfg.services.pdf = struct('mean_s',0.00186,'worst_s',0.00355);
cfg.services.quality_gate = struct('mean_s',0.10,'worst_s',0.20);
cfg.services.capture = struct('mean_s',1.5,'worst_s',2.5);

cfg.resources = struct();
cfg.resources.acquisition_workers = 1;
cfg.resources.upload_workers = 1;
cfg.resources.classifier_workers = 2;
cfg.resources.lesion_workers = 2;
cfg.resources.vessel_workers = 1;
cfg.resources.localization_workers = 2;
cfg.resources.xai_workers = 2;
cfg.resources.report_workers = 2;
cfg.resources.reviewers = 2;

% Optimization settings. Relative cost is intentionally not financial cost.
cfg.optimization = struct();
cfg.optimization.target_annual_patients = 100000;
cfg.optimization.max_queue_growth = 0;
cfg.optimization.max_p95_latency_minutes = 24 * 60;
cfg.optimization.bandwidth_options_mbps = [1 5 10 25 50];
cfg.optimization.vessel_worker_options = [1 2 4 8 16];
cfg.optimization.reviewer_options = [1 2 4 8 16];
cfg.optimization.classifier_worker_options = [1 2 4];
cfg.optimization.lesion_worker_options = [1 2 4];
cfg.optimization.xai_worker_options = [1 2 4];

cfg = local_merge(cfg, overrides);
cfg = local_derive(cfg);
end

function cfg = local_derive(cfg)
cfg.simulation_seconds = cfg.simulation_days * 24 * 3600;
cfg.required_rate_per_hour = cfg.annual_patients / (cfg.days_per_year * 24);
cfg.arrival_rate_per_hour = cfg.required_rate_per_hour * cfg.peak_multiplier;
cfg.arrival_rate_per_second = cfg.arrival_rate_per_hour / 3600;
cfg.interarrival_seconds = 1 / max(cfg.arrival_rate_per_second, eps);
cfg.network_latency_s = cfg.network_latency_ms / 1000;
cfg.transmission_time_s = (cfg.image_size_mb * 8) / max(cfg.bandwidth_mbps, eps) + cfg.network_latency_s;
if strcmpi(cfg.timing_mode, 'WORST_CASE')
    cfg.active_service_times = struct( ...
        'classification', cfg.services.classification.worst_s, ...
        'lesion', cfg.services.lesion.worst_s, ...
        'localization', cfg.services.localization.worst_s, ...
        'vessel', cfg.services.vessel.worst_s, ...
        'xai_agreement', cfg.services.xai_agreement.worst_s, ...
        'retina_guard', cfg.services.retina_guard.worst_s, ...
        'pdf', cfg.services.pdf.worst_s, ...
        'capture', cfg.services.capture.worst_s, ...
        'quality_gate', cfg.services.quality_gate.worst_s);
else
    cfg.active_service_times = struct( ...
        'classification', cfg.services.classification.mean_s, ...
        'lesion', cfg.services.lesion.mean_s, ...
        'localization', cfg.services.localization.mean_s, ...
        'vessel', cfg.services.vessel.mean_s, ...
        'xai_agreement', cfg.services.xai_agreement.mean_s, ...
        'retina_guard', cfg.services.retina_guard.mean_s, ...
        'pdf', cfg.services.pdf.mean_s, ...
        'capture', cfg.services.capture.mean_s, ...
        'quality_gate', cfg.services.quality_gate.mean_s);
end
end

function result = local_merge(base, override)
result = base;
names = fieldnames(override);
for i = 1:numel(names)
    name = names{i};
    if isstruct(override.(name)) && isfield(base, name) && isstruct(base.(name))
        result.(name) = local_merge(base.(name), override.(name));
    else
        result.(name) = override.(name);
    end
end
end
