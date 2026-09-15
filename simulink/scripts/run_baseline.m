function result = run_baseline()
%RUN_BASELINE Build/update/run the baseline model and save measured outputs.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(root));
if ~exist(fullfile(root,'RETINA_NEXUS_SYSTEM.slx'),'file'), build_model(); end
cfg=scenario_baseline(); assignin('base','cfg',cfg);
load_system(fullfile(root,'RETINA_NEXUS_SYSTEM.slx'));
set_param('RETINA_NEXUS_SYSTEM','SimulationCommand','update');
sim('RETINA_NEXUS_SYSTEM','ReturnWorkspaceOutputs','on');
result=run_capacity_simulation(cfg,true); result.model_execution='PASS';
save(fullfile(root,'results','baseline_result.mat'),'result','cfg','-v7.3');
write_json_file(compact_capacity_result(result),fullfile(root,'results','baseline_result.json'));
close_system('RETINA_NEXUS_SYSTEM',0);
fprintf('BASELINE_SIMULATION_PASS generated=%d completed=%d bottleneck=%s\n', ...
    result.counts.generated,result.counts.patient_outcomes_completed,result.bottleneck.stage);
end
