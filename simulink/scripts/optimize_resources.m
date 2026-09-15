function [best,optimizationTable]=optimize_resources()
%OPTIMIZE_RESOURCES Find minimum relative-cost configuration meeting target.
% Cost is a normalized engineering score, not a financial estimate.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(root));
base=retina_nexus_sim_config(); o=base.optimization;
rows=zeros(numel(o.bandwidth_options_mbps)*numel(o.classifier_worker_options)*numel(o.lesion_worker_options)*numel(o.vessel_worker_options)*numel(o.xai_worker_options)*numel(o.reviewer_options),13); k=0;
for bw=o.bandwidth_options_mbps
 for cw=o.classifier_worker_options
  for lw=o.lesion_worker_options
   for vw=o.vessel_worker_options
    for xw=o.xai_worker_options
     for rw=o.reviewer_options
      % A one-day deterministic horizon keeps the exhaustive candidate
      % search practical; annual sustainability is still decided from the
      % same analytical capacity equations used by the event runner.
      c=retina_nexus_sim_config(struct('scenario_name','OPTIMIZATION_CANDIDATE','bandwidth_mbps',bw,'simulation_days',1, ...
          'arrival_process','DETERMINISTIC', ...
          'resources',struct('classifier_workers',cw,'lesion_workers',lw,'vessel_workers',vw,'xai_workers',xw,'reviewers',rw)));
      q=run_capacity_simulation(c,false); k=k+1; cost=bw+cw+lw+vw+xw+rw;
      feasible=q.sustainability.analytical_capacity_patients_per_year>=o.target_annual_patients && q.bottleneck.offered_utilization<=1 && q.latency.full_report_seconds.p95_seconds<=o.max_p95_latency_minutes*60;
      rows(k,:)=[bw,cw,lw,vw,xw,rw,cost,q.sustainability.analytical_capacity_patients_per_year,q.bottleneck.offered_utilization,q.latency.full_report_seconds.p95_seconds,double(feasible),q.pending.evidence,q.pending.review];
     end
    end
   end
  end
 end
end
optimizationTable=array2table(rows,'VariableNames',{'BandwidthMbps','ClassifierWorkers','LesionWorkers','VesselWorkers','XAIWorkers','Reviewers','RelativeResourceCost','AnalyticalCapacity','BottleneckOfferedUtilization','P95LatencySeconds','Feasible','EvidencePending','ReviewPending'});
feasible=optimizationTable(optimizationTable.Feasible>0,:);
if isempty(feasible)
    [~,idx]=min(optimizationTable.RelativeResourceCost); best=optimizationTable(idx,:); best.Feasible=false;
else
    [~,idx]=min(feasible.RelativeResourceCost); best=feasible(idx,:);
end
out=fullfile(root,'results'); if ~exist(out,'dir'), mkdir(out); end
writetable(optimizationTable,fullfile(out,'resource_optimization.csv')); save(fullfile(out,'resource_optimization.mat'),'optimizationTable','best','-v7.3');
write_json_file(struct('recommended_configuration',table2struct(best),'feasible_candidates',height(feasible), ...
    'target_patients_per_year',o.target_annual_patients,'relative_cost_definition','Bandwidth tier plus worker counts; not financial cost'), ...
    fullfile(out,'resource_optimization.json'));
fprintf('OPTIMIZATION_PASS recommended_cost=%g capacity=%g bottleneck_utilization=%g\n', ...
    best.RelativeResourceCost,best.AnalyticalCapacity,best.BottleneckOfferedUtilization);
end
