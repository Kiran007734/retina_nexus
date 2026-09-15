function sweepTable=run_parameter_sweep()
%RUN_PARAMETER_SWEEP Practical resource/bandwidth/workload sweep.
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(root));
bandwidth=[1 5 10 25 50]; vessels=[1 2 4 8 16]; reviewers=[1 2 4 8 16]; workloads=[100000 125000 150000 200000];
rows=zeros(numel(bandwidth)*numel(vessels)*numel(reviewers)*numel(workloads),12); k=0;
for a=1:numel(workloads)
    for b=1:numel(bandwidth)
        for v=1:numel(vessels)
            for r=1:numel(reviewers)
                c=retina_nexus_sim_config(struct('scenario_name','PARAMETER_SWEEP', ...
                    'annual_patients',workloads(a),'bandwidth_mbps',bandwidth(b), ...
                    'simulation_days',2,'resources',struct('vessel_workers',vessels(v),'reviewers',reviewers(r))));
                q=run_capacity_simulation(c,false); k=k+1;
                rows(k,:)=[workloads(a),bandwidth(b),vessels(v),reviewers(r),q.throughput.annualized_completed_patients, ...
                    q.sustainability.analytical_capacity_patients_per_year,q.latency.full_report_seconds.p95_seconds, ...
                    q.bottleneck.offered_utilization,double(q.sustainability.sustainable),q.pending.evidence,q.pending.review,q.counts.generated];
            end
        end
    end
end
sweepTable=array2table(rows,'VariableNames',{'AnnualPatients','BandwidthMbps','VesselWorkers','Reviewers', ...
    'AnnualizedCompleted','AnalyticalCapacity','P95LatencySeconds','BottleneckOfferedUtilization', ...
    'Sustainable','EvidencePending','ReviewPending','Generated'});
out=fullfile(root,'results'); if ~exist(out,'dir'), mkdir(out); end
writetable(sweepTable,fullfile(out,'parameter_sweep.csv')); save(fullfile(out,'parameter_sweep.mat'),'sweepTable','-v7.3');
write_json_file(table2struct(sweepTable),fullfile(out,'parameter_sweep.json'));
fprintf('PARAMETER_SWEEP_PASS rows=%d\n',height(sweepTable));
end
