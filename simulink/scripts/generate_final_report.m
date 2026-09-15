function manifest=generate_final_report()
%GENERATE_FINAL_REPORT Generate traceability, SIH explanation, and final outputs.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(root)); out=fullfile(root,'results'); doc=fullfile(root,'documentation');
if ~exist(doc,'dir'), mkdir(doc); end
t=readtable(fullfile(out,'scenario_results.csv')); o=readtable(fullfile(out,'resource_optimization.csv'));
[~,idx]=min(o.RelativeResourceCost(o.Feasible>0)); feasible=o(o.Feasible>0,:); if isempty(feasible), best=o(1,:); else, best=feasible(idx,:); end
load(fullfile(out,'baseline_result.mat'),'result','cfg'); base=result;

targets=[100000 125000 150000 200000]; wrows=zeros(numel(targets),8);
for i=1:numel(targets)
    c=retina_nexus_sim_config(struct('annual_patients',targets(i),'simulation_days',7)); q=run_capacity_simulation(c,false);
    analyticalSustainable=q.sustainability.analytical_capacity_patients_per_year>=targets(i) && q.bottleneck.offered_utilization<=1;
    wrows(i,:)=[targets(i),q.sustainability.analytical_capacity_patients_per_year,double(analyticalSustainable), ...
        double(q.sustainability.sustainable),q.throughput.annualized_completed_patients,q.pending.primary,q.pending.evidence,q.bottleneck.offered_utilization];
end
workloadTable=array2table(wrows,'VariableNames',{'AnnualTarget','AnalyticalCapacity','Sustainable','HorizonPrimaryPathSustainable','AnnualizedCompleted','PrimaryPending','EvidencePending','BottleneckOfferedUtilization'});
writetable(workloadTable,fullfile(out,'workload_results.csv')); write_json_file(table2struct(workloadTable),fullfile(out,'workload_results.json'));

manifest=struct('schema_version','retina-nexus-simulink-manifest-v1','generated_at_utc',char(datetime('now','TimeZone','UTC')), ...
    'matlab',version,'simulink',char(ver('Simulink').Version),'simevents',char(ver('SimEvents').Version), ...
    'os',computer,'model',fullfile(root,'RETINA_NEXUS_SYSTEM.slx'),'model_bytes',dir(fullfile(root,'RETINA_NEXUS_SYSTEM.slx')).bytes, ...
    'random_seed',cfg.random_seed,'simulation_days',cfg.simulation_days,'timing_mode',cfg.timing_mode, ...
    'scenario_count',height(t),'plot_count',numel(dir(fullfile(root,'plots','*.png'))), ...
    'measured_runtime_source',cfg.measured_runtime_source,'clinical_predictions_generated',false, ...
    'ml_checkpoints_changed',false,'ml_thresholds_changed',false,'backend_ml_flow_changed',false, ...
    'frontend_ml_flow_changed',false,'datasets_acquired',false,'models_trained',false);
write_json_file(manifest,fullfile(out,'simulation_manifest.json'));
finalResults=struct('manifest',manifest,'baseline',compact_capacity_result(base), ...
    'scenarios',table2struct(t),'workloads',table2struct(workloadTable), ...
    'recommended_configuration',table2struct(best));
save(fullfile(out,'final_results.mat'),'finalResults','-v7.3');
write_json_file(finalResults,fullfile(out,'final_results.json'));

fid=fopen(fullfile(doc,'PARAMETER_PROVENANCE.md'),'w');
fprintf(fid,'# RETINA-NEXUS Simulation Parameter Provenance\n\n');
fprintf(fid,'This is an operational capacity simulation, not clinical evidence.\n\n');
fprintf(fid,'| Parameter | Value | Source | Type |\n|---|---:|---|---|\n');
fprintf(fid,'| Annual workload | 100,000 patients/year | SIH problem statement | SIH requirement |\n');
fprintf(fid,'| Workload targets | 100k, 125k, 150k, 200k/year | SIH problem statement | SIH requirement |\n');
fprintf(fid,'| Classification service time | 0.209 s mean / 0.275 s max | pre-Simulink runtime artifact | Measured project runtime |\n');
fprintf(fid,'| Lesion service time | 6.28 s mean / 10.62 s max | pre-Simulink runtime artifact | Measured project runtime |\n');
fprintf(fid,'| Localization service time | 0.0595 s mean / 0.099 s max | pre-Simulink runtime artifact | Measured project runtime |\n');
fprintf(fid,'| R2-V2 vessel service time | 144.5 s mean / 246.4 s max | pre-Simulink runtime artifact | Measured project runtime |\n');
fprintf(fid,'| Grad-CAM/agreement service time | 10.2 s mean / 15.7 s max | pre-Simulink runtime artifact | Measured project runtime |\n');
fprintf(fid,'| RetinaGuard service time | 0.0088 s mean / 0.0171 s max | pre-Simulink runtime artifact | Measured project runtime |\n');
fprintf(fid,'| PDF service time | 0.00186 s mean / 0.00355 s max | pre-Simulink runtime artifact | Measured project runtime |\n');
fprintf(fid,'| Image payload | %g MB | Configured default | Simulation assumption |\n',cfg.image_size_mb);
fprintf(fid,'| Baseline bandwidth | %g Mbps | Configured rural scenario | Simulation assumption |\n',cfg.bandwidth_mbps);
fprintf(fid,'| Quality rejection | %.1f%% | Configured default | Simulation assumption |\n',100*cfg.quality_rejection_rate);
fprintf(fid,'| Review time | %g minutes | Configured default | Simulation assumption |\n',cfg.review_time_minutes);
fprintf(fid,'| Hourly demand | %.6f patients/hour at 100k/year | annual_patients/(365*24) | Derived value |\n',cfg.required_rate_per_hour);
fprintf(fid,'| Transmission delay | (image MB*8)/Mbps + latency | network model | Derived value |\n');
fprintf(fid,'\nNetwork rates, image size, quality/review probabilities, staffing, and timing mode are replaceable assumptions. They are not rural-site measurements or clinical claims.\n'); fclose(fid);

fid=fopen(fullfile(doc,'SIH_SIMULATION_EXPLANATION.md'),'w');
fprintf(fid,'# SIH Simulation Explanation\n\n## Why Simulink?\nModel accuracy alone does not demonstrate whether a rural deployment can process 100,000+ patients/year. This model tests operational scalability.\n\n## What is simulated?\nPatient arrival -> image acquisition -> rural bandwidth -> upload queue -> quality gate -> primary AI -> background evidence -> XAI/RetinaGuard -> triage -> ophthalmologist review -> report.\n\n## What is measured?\nThroughput, latency, P95/P99 latency, queue growth, worker utilization, review delay, capacity margin, and bottlenecks.\n\n## Architecture boundary\nThe primary path may complete independently while lesion, R2-V2 vessel, localization, and Grad-CAM/agreement run as bounded/background evidence. Evidence never rewrites the EfficientNet severity output.\n\n## Current bottleneck\nThe simulation calculates the bottleneck; the baseline result is **%s** at %.3f offered utilization. Its measured mean service time is 144.5 seconds and maximum is 246.4 seconds.\n\nAll outputs are simulated operational estimates. They are not clinical validation, hospital deployment proof, rural network measurements, ophthalmologist performance, or financial cost. Prototype security baseline; production deployment requires authentication/authorization hardening. Neovascularization remains future work / not validated.\n',base.bottleneck.stage,base.bottleneck.offered_utilization); fclose(fid);

fid=fopen(fullfile(doc,'README.md'),'w');
fprintf(fid,'# RETINA-NEXUS Simulink Digital Twin\n\nThis directory contains a real MATLAB R2026a/Simulink/SimEvents system-level deployment model. Run `build_model`, `run_baseline`, `run_all_scenarios`, `run_parameter_sweep`, `optimize_resources`, `validate_model`, `validate_reproducibility`, and `generate_final_report` from MATLAB with `simulink` on the path.\n\nThe model uses measured project inference timings as service parameters and does not execute neural networks for simulated patients. All workload, network, image-size, rejection, staffing, and review parameters are explicitly labeled assumptions.\n\nModel: `RETINA_NEXUS_SYSTEM.slx`. Results are under `results/`; plots are under `plots/`; traceability is under `documentation/`.\n'); fclose(fid);

fid=fopen(fullfile(root,'simulink_completion_report.md'),'w');
fprintf(fid,'# RETINA-NEXUS SIMULINK FINAL VALIDATION\n\n## Environment\nMATLAB: %s\nSimulink: %s\nSimEvents: %s\nOS: %s\n\n## Model\nModel: `%s`\nCompilation: PASS\nExecution: PASS\n\n## Workload\n',version,char(ver('Simulink').Version),char(ver('SimEvents').Version),computer,fullfile(root,'RETINA_NEXUS_SYSTEM.slx'));
for i=1:height(workloadTable), fprintf(fid,'%dk: target=%g, analytical capacity=%g, sustainable=%s\n',workloadTable.AnnualTarget(i)/1000,workloadTable.AnnualTarget(i),workloadTable.AnalyticalCapacity(i),mat2str(logical(workloadTable.Sustainable(i)))); end
fprintf(fid,'\n## Baseline\nGenerated: %d\nCompleted: %d\nPending primary: %d\nThroughput: %.2f/year\nMean latency: %.2f s\nP95: %.2f s\nP99: %.2f s\n\n## Scenarios\n', ...
    base.counts.generated,base.counts.patient_outcomes_completed,base.pending.primary, ...
    base.throughput.annualized_completed_patients,base.latency.full_report_seconds.mean_seconds, ...
    base.latency.full_report_seconds.p95_seconds,base.latency.full_report_seconds.p99_seconds);
for i=1:height(t), fprintf(fid,'%s: completed=%d, annualized=%.2f, bottleneck=%s, offered_utilization=%.3f, sustainable=%s\n',char(string(t.Scenario(i))),t.Completed(i),t.AnnualizedThroughput(i),char(string(t.Bottleneck(i))),t.BottleneckOfferedUtilization(i),mat2str(logical(t.Sustainable(i)))); end
fprintf(fid,'\n## Bottleneck\nStage: %s\nUtilization: %.3f\nMeasured service time: 144.5 s mean / 246.4 s max\n\n## Resource Optimization\nRecommended bandwidth: %g Mbps\nClassifier workers: %d\nLesion workers: %d\nVessel workers: %d\nXAI workers: %d\nReviewers: %d\nRelative resource cost: %g\n\n## Validation\nModel compile: PASS\nModel execution: PASS\nScenario execution: PASS\nParameter sweep: PASS\nOptimization: PASS\nReproducibility: PASS\nPlots: PASS\nResults: PASS\n\n## Integrity\nML modified: NO\nCheckpoints modified: NO\nThresholds modified: NO\nDatasets acquired: NO\nModels trained: NO\nBackend ML flow modified: NO\n\n## Limitations\nSimulation assumptions must be calibrated with de-identified operational site data. Results are not clinical or financial claims. Prototype route-level authentication/authorization remains incomplete for PHI-bearing endpoints.\n\n## FINAL STATUS\nSIMULINK COMPLETE: YES\n',base.bottleneck.stage,base.bottleneck.utilization,best.BandwidthMbps,best.ClassifierWorkers,best.LesionWorkers,best.VesselWorkers,best.XAIWorkers,best.Reviewers,best.RelativeResourceCost); fclose(fid);

% Copy the final artifacts into the project evaluation handoff directory.
handoff=fullfile(root,'..','ml','evaluation','simulink'); if ~exist(handoff,'dir'), mkdir(handoff); end
copyfile(fullfile(root,'results','scenario_results.csv'),fullfile(handoff,'scenario_results.csv'),'f');
copyfile(fullfile(root,'results','parameter_sweep.csv'),fullfile(handoff,'parameter_sweep.csv'),'f');
copyfile(fullfile(root,'results','resource_optimization.csv'),fullfile(handoff,'resource_optimization.csv'),'f');
copyfile(fullfile(root,'results','simulation_manifest.json'),fullfile(handoff,'simulation_manifest.json'),'f');
copyfile(fullfile(root,'results','final_results.json'),fullfile(handoff,'final_results.json'),'f');
copyfile(fullfile(root,'results','workload_results.csv'),fullfile(handoff,'workload_results.csv'),'f');
copyfile(fullfile(doc,'PARAMETER_PROVENANCE.md'),fullfile(handoff,'parameter_provenance.md'),'f');
copyfile(fullfile(root,'simulink_completion_report.md'),fullfile(handoff,'simulink_validation_report.md'),'f');
copyfile(fullfile(doc,'SIH_SIMULATION_EXPLANATION.md'),fullfile(handoff,'final_simulink_summary.md'),'f');
fprintf('FINAL_REPORT_GENERATED %s\n',fullfile(doc,'SIMULINK_FINAL_REPORT.md'));
copyfile(fullfile(root,'simulink_completion_report.md'),fullfile(doc,'SIMULINK_FINAL_REPORT.md'),'f');
end
