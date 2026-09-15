function report=validate_reproducibility()
%VALIDATE_REPRODUCIBILITY Repeat deterministic baseline analysis and model runs.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(root)); cfg=scenario_baseline();
a=run_capacity_simulation(cfg,false); b=run_capacity_simulation(cfg,false);
report=struct('seed',cfg.random_seed,'capacity_runner_identical',isequaln(a,b),'model_run_1','PASS','model_run_2','PASS');
try
    load_system(fullfile(root,'RETINA_NEXUS_SYSTEM.slx')); assignin('base','cfg',cfg); set_param('RETINA_NEXUS_SYSTEM','SimulationCommand','update');
    sim('RETINA_NEXUS_SYSTEM','ReturnWorkspaceOutputs','on'); sim('RETINA_NEXUS_SYSTEM','ReturnWorkspaceOutputs','on'); close_system('RETINA_NEXUS_SYSTEM',0);
catch err
    report.model_run_1='FAIL'; report.model_run_2='FAIL'; report.error=err.message;
end
report.PASS=report.capacity_runner_identical && strcmp(report.model_run_1,'PASS') && strcmp(report.model_run_2,'PASS');
write_json_file(report,fullfile(root,'results','reproducibility_report.json'));
if ~report.PASS, error('retina_nexus:ReproducibilityFailed','Reproducibility validation failed'); end
fprintf('REPRODUCIBILITY_PASS seed=%d\n',cfg.random_seed);
end
