function modelPath = build_model()
%BUILD_MODEL Build the real RETINA-NEXUS SimEvents deployment twin.
% The model is intentionally operational: measured ML service times are
% parameters, while the neural networks remain outside Simulink.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(root));
modelName = 'RETINA_NEXUS_SYSTEM';
modelPath = fullfile(root,[modelName '.slx']);
cfg = scenario_baseline();
assignin('base','cfg',cfg);

if bdIsLoaded(modelName), close_system(modelName,0); end
if exist(modelPath,'file'), delete(modelPath); end
new_system(modelName);
open_system(modelName);
set_param(modelName,'Solver','FixedStepDiscrete','FixedStep','1','StopTime','cfg.simulation_seconds');
set_param(modelName,'ModelBrowserVisibility','on');

% Real SimEvents workflow blocks.
add_block('sldelib/Entity Generator',[modelName '/PATIENT_ARRIVAL'], ...
    'Position',[30 250 170 320]);
set_param([modelName '/PATIENT_ARRIVAL'],'GenerationMethod','Time-based', ...
    'TimeSource','Dialog','Period','cfg.interarrival_seconds', ...
    'GenerateEntityAtSimulationStart','off','EntityType','Structured', ...
    'AttributeName','QualityPort', ...
    'AttributeInitialValue','1+(rand(1,1)<cfg.quality_rejection_rate)');

add_block('sldelib/Entity Queue',[modelName '/UPLOAD_QUEUE'], ...
    'Position',[220 250 350 320]);
set_param([modelName '/UPLOAD_QUEUE'],'Capacity','inf','QueueType','FIFO');
add_block('sldelib/Entity Server',[modelName '/RURAL_NETWORK'], ...
    'Position',[400 250 540 320]);
set_param([modelName '/RURAL_NETWORK'],'Capacity','cfg.resources.upload_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.transmission_time_s');
add_block('sldelib/Entity Server',[modelName '/QUALITY_GATE'], ...
    'Position',[590 250 730 320]);
set_param([modelName '/QUALITY_GATE'],'Capacity','cfg.resources.acquisition_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.active_service_times.quality_gate');
add_block('sldelib/Entity Output Switch',[modelName '/QUALITY_GATE_ROUTER'], ...
    'Position',[740 250 780 320]);
set_param([modelName '/QUALITY_GATE_ROUTER'],'NumberOutputPorts','2', ...
    'SwitchingCriterion','From attribute','SwitchAttributeName','QualityPort', ...
    'InitialPortSelection','1');
add_block('sldelib/Entity Server',[modelName '/QUALITY_RECAPTURE_REJECT'], ...
    'Position',[800 360 950 430]);
set_param([modelName '/QUALITY_RECAPTURE_REJECT'],'Capacity','cfg.resources.acquisition_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.active_service_times.capture');
add_block('sldelib/Entity Terminator',[modelName '/QUALITY_REJECTED'], ...
    'Position',[1000 360 1130 430]);
add_block('sldelib/Entity Queue',[modelName '/PRIMARY_AI_QUEUE'], ...
    'Position',[830 250 960 320]);
set_param([modelName '/PRIMARY_AI_QUEUE'],'Capacity','inf','QueueType','FIFO');
add_block('sldelib/Entity Server',[modelName '/PRIMARY_AI'], ...
    'Position',[1010 250 1150 320]);
set_param([modelName '/PRIMARY_AI'],'Capacity','cfg.resources.classifier_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.active_service_times.classification');
add_block('sldelib/Entity Replicator',[modelName '/BACKGROUND_EVIDENCE_FORK'], ...
    'Position',[1200 250 1350 320]);
set_param([modelName '/BACKGROUND_EVIDENCE_FORK'],'ReplicasDepartFrom','Separate output ports', ...
    'ReplicationAmountSource','Dialog','NumberReplicas','1');

% Primary screening branch: classification -> trust -> triage -> human review -> report.
add_block('sldelib/Entity Server',[modelName '/XAI_RETINAGUARD'], ...
    'Position',[1360 120 1500 190]);
set_param([modelName '/XAI_RETINAGUARD'],'Capacity','cfg.resources.classifier_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.active_service_times.retina_guard');
add_block('sldelib/Entity Queue',[modelName '/HUMAN_REVIEW_QUEUE'], ...
    'Position',[1550 120 1690 190]);
set_param([modelName '/HUMAN_REVIEW_QUEUE'],'Capacity','inf','QueueType','FIFO');
add_block('sldelib/Entity Server',[modelName '/HUMAN_REVIEW'], ...
    'Position',[1740 120 1880 190]);
set_param([modelName '/HUMAN_REVIEW'],'Capacity','cfg.resources.reviewers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.review_time_minutes*60');
add_block('sldelib/Entity Server',[modelName '/REPORTING'], ...
    'Position',[1930 120 2070 190]);
set_param([modelName '/REPORTING'],'Capacity','cfg.resources.report_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.active_service_times.pdf', ...
    'NumberEntitiesDeparted','on');
add_block('sldelib/Entity Terminator',[modelName '/FINAL_RESULT'], ...
    'Position',[2120 120 2250 190]);
add_block('simulink/Sinks/Terminator',[modelName '/REPORT_METRIC_TERMINATOR'], ...
    'Position',[2120 210 2250 240]);

% Background evidence branch: lesion -> R2-V2 vessel -> localization -> XAI/agreement.
add_block('sldelib/Entity Queue',[modelName '/EVIDENCE_QUEUE'], ...
    'Position',[1360 430 1500 500]);
set_param([modelName '/EVIDENCE_QUEUE'],'Capacity','inf','QueueType','FIFO');
add_block('sldelib/Entity Server',[modelName '/LESION_EVIDENCE'], ...
    'Position',[1550 430 1690 500]);
set_param([modelName '/LESION_EVIDENCE'],'Capacity','cfg.resources.lesion_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.active_service_times.lesion');
add_block('sldelib/Entity Queue',[modelName '/VESSEL_QUEUE'], ...
    'Position',[1740 430 1880 500]);
set_param([modelName '/VESSEL_QUEUE'],'Capacity','inf','QueueType','FIFO');
add_block('sldelib/Entity Server',[modelName '/VESSEL_R2V2'], ...
    'Position',[1930 430 2070 500]);
set_param([modelName '/VESSEL_R2V2'],'Capacity','cfg.resources.vessel_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.active_service_times.vessel', ...
    'NumberEntitiesDeparted','on');
add_block('simulink/Sinks/Terminator',[modelName '/VESSEL_METRIC_TERMINATOR'], ...
    'Position',[2120 520 2250 550]);
add_block('sldelib/Entity Server',[modelName '/LOCALIZATION'], ...
    'Position',[2120 430 2260 500]);
set_param([modelName '/LOCALIZATION'],'Capacity','cfg.resources.localization_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.active_service_times.localization');
add_block('sldelib/Entity Server',[modelName '/XAI_AGREEMENT'], ...
    'Position',[2310 430 2450 500]);
set_param([modelName '/XAI_AGREEMENT'],'Capacity','cfg.resources.xai_workers', ...
    'ServiceTimeSource','Dialog','ServiceTimeValue','cfg.active_service_times.xai_agreement');
add_block('sldelib/Entity Terminator',[modelName '/EVIDENCE_COMPLETE'], ...
    'Position',[2500 430 2630 500]);

% Entity connections. Port 1 is the entity port; metric ports are terminated.
local_line(modelName,'PATIENT_ARRIVAL/1','UPLOAD_QUEUE/1');
local_line(modelName,'UPLOAD_QUEUE/1','RURAL_NETWORK/1');
local_line(modelName,'RURAL_NETWORK/1','QUALITY_GATE/1');
local_line(modelName,'QUALITY_GATE/1','QUALITY_GATE_ROUTER/1');
local_line(modelName,'QUALITY_GATE_ROUTER/1','PRIMARY_AI_QUEUE/1');
local_line(modelName,'QUALITY_GATE_ROUTER/2','QUALITY_RECAPTURE_REJECT/1');
local_line(modelName,'QUALITY_RECAPTURE_REJECT/1','QUALITY_REJECTED/1');
local_line(modelName,'PRIMARY_AI_QUEUE/1','PRIMARY_AI/1');
local_line(modelName,'PRIMARY_AI/1','BACKGROUND_EVIDENCE_FORK/1');
local_line(modelName,'BACKGROUND_EVIDENCE_FORK/1','XAI_RETINAGUARD/1');
local_line(modelName,'XAI_RETINAGUARD/1','HUMAN_REVIEW_QUEUE/1');
local_line(modelName,'HUMAN_REVIEW_QUEUE/1','HUMAN_REVIEW/1');
local_line(modelName,'HUMAN_REVIEW/1','REPORTING/1');
local_line(modelName,'REPORTING/2','FINAL_RESULT/1');
local_line(modelName,'BACKGROUND_EVIDENCE_FORK/2','EVIDENCE_QUEUE/1');
local_line(modelName,'EVIDENCE_QUEUE/1','LESION_EVIDENCE/1');
local_line(modelName,'LESION_EVIDENCE/1','VESSEL_QUEUE/1');
local_line(modelName,'VESSEL_QUEUE/1','VESSEL_R2V2/1');
local_line(modelName,'VESSEL_R2V2/2','LOCALIZATION/1');
local_line(modelName,'LOCALIZATION/1','XAI_AGREEMENT/1');
local_line(modelName,'XAI_AGREEMENT/1','EVIDENCE_COMPLETE/1');
local_line(modelName,'REPORTING/1','REPORT_METRIC_TERMINATOR/1');
local_line(modelName,'VESSEL_R2V2/1','VESSEL_METRIC_TERMINATOR/1');

% Documentation annotations make the primary/background boundary explicit.
add_block('built-in/Note',[modelName '/FLOW_NOTE'],'Position',[30 30 760 115]);
set_param([modelName '/FLOW_NOTE'],'Text','RETINA-NEXUS SYSTEM-LEVEL DIGITAL TWIN\nPrimary screening is independent of optional/background evidence.\nAll service times are measured project runtime parameters; workload/network/review values are simulation assumptions.');
add_block('built-in/Note',[modelName '/EVIDENCE_NOTE'],'Position',[1320 570 2490 650]);
set_param([modelName '/EVIDENCE_NOTE'],'Text','BACKGROUND EVIDENCE: Lesion -> R2-V2 vessel -> localization -> Grad-CAM/agreement\nR2-V2 remains visible as the measured CPU bottleneck. Evidence never rewrites disease severity.');
add_block('built-in/Note',[modelName '/INTEGRITY_NOTE'],'Position',[780 30 1500 115]);
set_param([modelName '/INTEGRITY_NOTE'],'Text','No neural networks execute per simulated patient.\nNo checkpoint, threshold, fusion, RetinaGuard, backend, frontend, or dataset behavior is changed.');

save_system(modelName,modelPath);
close_system(modelName,0);
fprintf('BUILT_MODEL=%s\n',modelPath);
end

function local_line(modelName,src,dst)
try
    add_line(modelName,src,dst,'autorouting','smart');
catch err
    error('retina_nexus:BuildConnection','Could not connect %s -> %s: %s',src,dst,err.message);
end
end
