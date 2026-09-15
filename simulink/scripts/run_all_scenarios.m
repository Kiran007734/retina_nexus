function [scenarioTable, results] = run_all_scenarios()
%RUN_ALL_SCENARIOS Execute the real model and capacity analysis for all scenarios.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(root));
if ~exist(fullfile(root,'RETINA_NEXUS_SYSTEM.slx'),'file'), build_model(); end
scenarioFns={@scenario_baseline,@scenario_poor_network,@scenario_peak_load,@scenario_compute_constrained, ...
    @scenario_vessel_scale,@scenario_review_scale,@scenario_optimized};
scenarioTable=table(); results=cell(numel(scenarioFns),1);
load_system(fullfile(root,'RETINA_NEXUS_SYSTEM.slx'));
for i=1:numel(scenarioFns)
    cfg=scenarioFns{i}(); assignin('base','cfg',cfg);
    set_param('RETINA_NEXUS_SYSTEM','SimulationCommand','update');
    sim('RETINA_NEXUS_SYSTEM','ReturnWorkspaceOutputs','on');
    results{i}=run_capacity_simulation(cfg,i==1); results{i}.model_execution='PASS';
    row=local_row(results{i});
    if isempty(scenarioTable)
        scenarioTable=row;
    else
        scenarioTable=[scenarioTable;row]; %#ok<AGROW>
    end
end
close_system('RETINA_NEXUS_SYSTEM',0);
writetable(scenarioTable,fullfile(root,'results','scenario_results.csv'));
save(fullfile(root,'results','scenario_results.mat'),'scenarioTable','results','-v7.3');
write_json_file(cellfun(@compact_capacity_result,results,'UniformOutput',false),fullfile(root,'results','scenario_results.json'));
generate_plots(scenarioTable,results,root);
fprintf('SCENARIOS_PASS count=%d\n',height(scenarioTable));
end

function row=local_row(x)
row=table(string(x.scenario),x.counts.generated,x.counts.patient_outcomes_completed, ...
    x.counts.ungradable_exits,x.counts.primary_completed,x.counts.evidence_completed, ...
    x.pending.primary,x.pending.evidence,x.latency.full_report_seconds.mean_seconds, ...
    x.latency.full_report_seconds.p95_seconds,x.latency.full_report_seconds.p99_seconds, ...
    x.throughput.annualized_completed_patients,string(x.bottleneck.stage),x.bottleneck.utilization, ...
    x.bottleneck.offered_utilization,x.sustainability.sustainable, ...
    'VariableNames',{'Scenario','Generated','Completed','UngradableExits','PrimaryCompleted', ...
    'EvidenceCompleted','PrimaryPending','EvidencePending','MeanLatencySeconds','P95LatencySeconds', ...
    'P99LatencySeconds','AnnualizedThroughput','Bottleneck','BottleneckUtilization', ...
    'BottleneckOfferedUtilization','Sustainable'});
end
