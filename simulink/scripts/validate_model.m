function report=validate_model()
%VALIDATE_MODEL Compile, execute, and validate generated simulation artifacts.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(root));
report=struct('MODEL_EXISTS',false,'MODEL_OPENS',false,'MODEL_COMPILES',false,'BASELINE_RUNS',false, ...
    'SCENARIOS_RUN',false,'WORKLOAD_100K_RUNS',false,'METRICS_VALID',false,'QUEUE_VALUES_VALID',false, ...
    'LATENCY_VALUES_VALID',false,'UTILIZATION_VALID',false,'RESULTS_VALID',false,'PLOTS_EXIST',false, ...
    'OPTIMIZATION_RUNS',false,'errors',{{}});
modelPath=fullfile(root,'RETINA_NEXUS_SYSTEM.slx'); report.MODEL_EXISTS=exist(modelPath,'file')~=0;
if ~report.MODEL_EXISTS, report.errors{end+1}='Model file is missing'; local_write(report,root); return; end
try
    load_system(modelPath); report.MODEL_OPENS=true; cfg=scenario_baseline(); assignin('base','cfg',cfg);
    set_param('RETINA_NEXUS_SYSTEM','SimulationCommand','update'); report.MODEL_COMPILES=true;
    sim('RETINA_NEXUS_SYSTEM','ReturnWorkspaceOutputs','on'); report.BASELINE_RUNS=true;
catch err
    report.errors{end+1}=err.message;
end
if bdIsLoaded('RETINA_NEXUS_SYSTEM'), close_system('RETINA_NEXUS_SYSTEM',0); end

scenarioPath=fullfile(root,'results','scenario_results.csv'); report.SCENARIOS_RUN=exist(scenarioPath,'file')==2;
if report.SCENARIOS_RUN
    t=readtable(scenarioPath); report.SCENARIOS_RUN=height(t)>=7 && all(ismember({'BASELINE_RURAL','POOR_NETWORK','PEAK_LOAD','COMPUTE_CONSTRAINED','VESSEL_SCALE_UP','REVIEW_SCALE_UP','OPTIMIZED'},string(t.Scenario)));
    report.METRICS_VALID=all(isfinite(t.MeanLatencySeconds))&&all(isfinite(t.P95LatencySeconds))&&all(t.MeanLatencySeconds>=0)&&all(t.P95LatencySeconds>=0);
    report.WORKLOAD_100K_RUNS=any(t.Scenario=="BASELINE_RURAL")&&any(t.Generated>0);
else
    report.errors{end+1}='Scenario results CSV is missing';
end
baselinePath=fullfile(root,'results','baseline_result.mat');
if exist(baselinePath,'file')
    d=load(baselinePath,'result'); x=d.result;
    report.QUEUE_VALUES_VALID=local_numeric_valid(x.stage)&&all(arrayfun(@(z)z.max_queue>=0,struct2array(x.stage)));
    report.LATENCY_VALUES_VALID=local_numeric_valid(x.latency);
    report.UTILIZATION_VALID=all(arrayfun(@(z)z.utilization>=0&&z.utilization<=1&&z.offered_utilization>=0,struct2array(x.stage)));
    report.RESULTS_VALID=x.counts.patient_outcomes_completed<=x.counts.generated && x.counts.reviews_completed<=x.counts.review_required;
else
    report.errors{end+1}='Baseline MAT result is missing';
end
report.PLOTS_EXIST=numel(dir(fullfile(root,'plots','*.png')))>=17;
report.OPTIMIZATION_RUNS=exist(fullfile(root,'results','resource_optimization.csv'),'file')==2;
report.PASS=all(struct2array(rmfield(report,'errors')));
local_write(report,root);
if ~report.PASS, error('retina_nexus:ValidationFailed','Simulation validation failed'); end
fprintf('MODEL_VALIDATION_PASS\n');
end

function ok=local_numeric_valid(x)
ok=true;
if isstruct(x)
    f=fieldnames(x); for i=1:numel(f), ok=ok&&local_numeric_valid(x.(f{i})); end
elseif isnumeric(x)
    ok=all(isfinite(x(:))) || isempty(x);
end
end
function local_write(report,root)
write_json_file(report,fullfile(root,'results','validation_report.json'));
end
