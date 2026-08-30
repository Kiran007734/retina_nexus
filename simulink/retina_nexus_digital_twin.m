function [summary, events] = retina_nexus_digital_twin(config, scenario)
%RETINA_NEXUS_DIGITAL_TWIN Operational capacity-planning reference model.
% This is not a clinical model and does not estimate diagnostic accuracy.
% Times are minutes. Requires base MATLAB only.

if nargin < 1 || isempty(config)
    config = struct();
end
if nargin < 2 || isempty(scenario)
    scenario = "NORMAL LOAD";
end

p = local_defaults();
p = local_apply_scenario(p, upper(string(scenario)));
p = local_merge(p, config);
rng(p.seed);

totalMinutes = p.simulation_days * 24 * 60;
arrivalProbability = min(0.99, p.patients_per_day / (24 * 60));
acquisitionAvailable = zeros(1, p.acquisition_server_count);
aiAvailable = zeros(1, p.ai_worker_count);
specialistAvailable = zeros(1, max(1, p.specialist_count));

rows = zeros(0, 12);
patientId = 0;
recaptureCount = 0;
ungradableCount = 0;
ungradableExitCount = 0;
aiCount = 0;
reviewCount = 0;
referralCount = 0;
completedCount = 0;

for minute = 0:(totalMinutes - 1)
    if rand() >= arrivalProbability
        continue;
    end
    patientId = patientId + 1;
    arrival = minute;
    attempts = 0;
    captured = false;
    ungradable = false;
    captureStart = arrival;
    captureEnd = arrival;
    qualityEnd = arrival;

    while ~captured
        attempts = attempts + 1;
        serverIndex = local_first_available(acquisitionAvailable);
        captureStart = max(arrival, acquisitionAvailable(serverIndex));
        captureEnd = captureStart + p.image_capture_time;
        qualityEnd = captureEnd + p.quality_gate_time;
        acquisitionAvailable(serverIndex) = captureEnd;
        ungradable = rand() < p.ungradable_rate;
        if ~ungradable
            captured = true;
        elseif attempts <= p.max_recaptures && rand() < p.recapture_rate
            recaptureCount = recaptureCount + 1;
            arrival = qualityEnd;
        else
            ungradableExitCount = ungradableExitCount + 1;
            ungradableCount = ungradableCount + 1;
            break;
        end
    end
    if ~captured
        rows(end + 1, :) = [patientId, minute, captureStart, captureEnd, qualityEnd, 1, attempts - 1, NaN, NaN, NaN, 0, 0]; %#ok<AGROW>
        continue;
    end
    if attempts > 1
        ungradableCount = ungradableCount + attempts - 1;
    end

    aiIndex = local_first_available(aiAvailable);
    aiStart = max(qualityEnd, aiAvailable(aiIndex));
    aiEnd = aiStart + p.ai_inference_time + p.bandwidth_delay;
    aiAvailable(aiIndex) = aiEnd;
    aiCount = aiCount + 1;

    referral = rand() < p.referral_rate;
    reviewRequired = referral || rand() < p.human_review_rate;
    reviewStart = NaN;
    reviewEnd = NaN;
    if reviewRequired
        specialistIndex = local_first_available(specialistAvailable);
        reviewStart = max(aiEnd, specialistAvailable(specialistIndex));
        reviewEnd = reviewStart + p.review_time;
        specialistAvailable(specialistIndex) = reviewEnd;
        reviewCount = reviewCount + 1;
    end
    referralCount = referralCount + referral;
    completedCount = completedCount + 1;
    rows(end + 1, :) = [patientId, minute, captureStart, captureEnd, qualityEnd, 0, attempts - 1, aiStart, aiEnd, reviewStart, reviewEnd, referral]; %#ok<AGROW>
end

events = array2table(rows, 'VariableNames', {'PatientId', 'ArrivalMinute', ...
    'CaptureStartMinute', 'CaptureEndMinute', 'QualityEndMinute', ...
    'Ungradable', 'RecaptureCount', 'AIStartMinute', 'AIEndMinute', ...
    'ReviewStartMinute', 'ReviewEndMinute', 'Referral'});

reviewWait = events.ReviewStartMinute - events.AIEndMinute;
reviewWait = reviewWait(isfinite(reviewWait));
specialistBusy = sum(events.ReviewEndMinute(isfinite(events.ReviewEndMinute)) - events.ReviewStartMinute(isfinite(events.ReviewStartMinute)));
acquisitionBusy = sum(events.CaptureEndMinute - events.CaptureStartMinute);
aiBusy = completedCount * p.ai_inference_time;
stageNames = {'acquisition', 'ai_processing', 'specialist_review'};
stageUtilization = [acquisitionBusy / max(1, totalMinutes * p.acquisition_server_count), ...
    aiBusy / max(1, totalMinutes * p.ai_worker_count), ...
    specialistBusy / max(1, totalMinutes * max(1, p.specialist_count))];

queueSamples = zeros(0, 1);
if ~isempty(events)
    sampleTimes = unique([events.AIEndMinute(isfinite(events.AIEndMinute)); events.ReviewStartMinute(isfinite(events.ReviewStartMinute))]);
    for t = sampleTimes'
        queueSamples(end + 1, 1) = sum(events.AIEndMinute <= t & events.ReviewStartMinute > t); %#ok<AGROW>
    end
end
maxQueue = max([0; queueSamples]);
targetUtilization = 0.80;
requiredSpecialists = ceil((reviewCount * p.review_time) / max(1, totalMinutes * targetUtilization));

[~, order] = sort(stageUtilization, 'descend');
bottlenecks = table(string(stageNames(order))', stageUtilization(order)', ...
    'VariableNames', {'Stage', 'Utilization'});
summary = struct();
summary.scenario = char(upper(string(scenario)));
summary.parameters = p;
summary.counts = struct('arrivals', patientId, 'gradable_ai_cases', aiCount, ...
    'ungradable_captures', ungradableCount, 'ungradable_exits', ungradableExitCount, ...
    'recaptures', recaptureCount, 'specialist_reviews', reviewCount, ...
    'referrals', referralCount, 'completed_outcomes', completedCount);
summary.throughput_per_day = completedCount / max(1, p.simulation_days);
summary.queue = struct('max_specialist_queue', maxQueue, ...
    'mean_specialist_waiting_minutes', local_mean(reviewWait), ...
    'mean_review_queue_length', local_mean(queueSamples));
summary.staff_utilization = struct('acquisition', stageUtilization(1), ...
    'ai_processing', stageUtilization(2), 'specialist_review', stageUtilization(3));
summary.bottlenecks = bottlenecks;
summary.resource_requirements = struct('specialists_at_target_utilization', ...
    max(1, requiredSpecialists), 'target_utilization', targetUtilization);
summary.note = 'Operational planning estimate only; not clinical evidence or a diagnostic simulation.';
end

function p = local_defaults()
p = struct('patients_per_day', 60, 'image_capture_time', 1.5, ...
    'ungradable_rate', 0.08, 'recapture_rate', 0.70, 'ai_inference_time', 0.20, ...
    'bandwidth_delay', 0.10, 'specialist_count', 2, 'review_time', 8.0, ...
    'referral_rate', 0.25, 'human_review_rate', 0.20, 'quality_gate_time', 0.05, ...
    'ai_worker_count', 2, 'acquisition_server_count', 1, ...
    'simulation_days', 1, 'seed', 42, 'max_recaptures', 2);
end

function p = local_apply_scenario(p, scenario)
switch scenario
    case "HIGH LOAD"
        p.patients_per_day = 180;
    case "LOW BANDWIDTH"
        p.bandwidth_delay = 4.0;
    case "HIGH UNGRADABLE RATE"
        p.ungradable_rate = 0.35;
        p.recapture_rate = 0.80;
    case "LIMITED SPECIALIST CAPACITY"
        p.specialist_count = 1;
        p.review_time = 10.0;
end
end

function p = local_merge(p, overrides)
names = fieldnames(overrides);
for index = 1:numel(names)
    if isfield(p, names{index})
        p.(names{index}) = overrides.(names{index});
    end
end
end

function index = local_first_available(values)
[~, index] = min(values);
end

function value = local_mean(values)
if isempty(values)
    value = 0;
else
    value = mean(values);
end
end
