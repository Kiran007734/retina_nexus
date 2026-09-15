function generate_plots(scenarioTable,results,root)
%GENERATE_PLOTS Generate operational, non-clinical SIH presentation plots.
plotDir=fullfile(root,'plots'); if ~exist(plotDir,'dir'), mkdir(plotDir); end
base=results{1};

f=local_fig(); t=base.trace.arrivals/3600; plot(t,(1:numel(t))','LineWidth',1.2); grid on; xlabel('Simulation time (hours)'); ylabel('Cumulative arrivals'); title('Patient throughput / arrival volume'); local_save(f,plotDir,'01_throughput_vs_time.png');
local_queue_plot(base,plotDir,'upload','02_upload_queue_vs_time.png','Upload queue length');
local_queue_plot(base,plotDir,'classifier','03_ai_queue_vs_time.png','Classifier queue length');
local_queue_plot(base,plotDir,'vessel','04_vessel_queue_vs_time.png','R2-V2 vessel queue length');
local_queue_plot(base,plotDir,'review','05_review_queue_vs_time.png','Human review queue length');

f=local_fig(); names=fieldnames(base.stage); util=zeros(numel(names),1); for i=1:numel(names), util(i)=base.stage.(names{i}).utilization; end; bar(util); ylim([0 1.05]); grid on; xticks(1:numel(names)); xticklabels(names); xtickangle(35); ylabel('Utilization'); title('Baseline resource utilization'); local_save(f,plotDir,'06_resource_utilization.png');
f=local_fig(); histogram(base.trace.full_latency/60,30); grid on; xlabel('Full report latency (minutes)'); ylabel('Cases'); title('Baseline end-to-end latency distribution'); local_save(f,plotDir,'07_latency_distribution.png');
f=local_fig(); bar(categorical(scenarioTable.Scenario),scenarioTable.P95LatencySeconds/60); grid on; ylabel('P95 latency (minutes)'); title('P95 latency by scenario'); local_save(f,plotDir,'08_p95_latency_scenarios.png');

band=[1 5 10 25 50]; tx=zeros(size(band)); for i=1:numel(band), tx(i)=5*8/band(i)+0.150; end
f=local_fig(); plot(band,tx,'-o','LineWidth',1.5); grid on; xlabel('Bandwidth (Mbps)'); ylabel('Transmission delay (s)'); title('Bandwidth vs transmission latency (assumption)'); local_save(f,plotDir,'09_bandwidth_vs_transmission_latency.png');
e2e=zeros(size(band)); for i=1:numel(band), c=scenario_baseline(); c.bandwidth_mbps=band(i); c=local_rederive(c); q=run_capacity_simulation(c,false); e2e(i)=q.latency.full_report_seconds.mean_seconds/60; end
f=local_fig(); plot(band,e2e,'-o','LineWidth',1.5); grid on; xlabel('Bandwidth (Mbps)'); ylabel('Mean full latency (minutes)'); title('Bandwidth vs end-to-end latency'); local_save(f,plotDir,'10_bandwidth_vs_end_to_end_latency.png');

vesselWorkers=[1 2 4 8 16]; vesselThroughput=zeros(size(vesselWorkers)); for i=1:numel(vesselWorkers), c=scenario_peak_load(); c.resources.vessel_workers=vesselWorkers(i); q=run_capacity_simulation(c,false); vesselThroughput(i)=q.throughput.annualized_completed_patients; end
f=local_fig(); plot(vesselWorkers,vesselThroughput,'-o','LineWidth',1.5); grid on; xlabel('Vessel workers'); ylabel('Annualized completed cases'); title('Vessel worker scaling'); local_save(f,plotDir,'11_vessel_workers_vs_throughput.png');
reviewers=[1 2 4 8 16]; reviewLatency=zeros(size(reviewers)); for i=1:numel(reviewers), c=scenario_peak_load(); c.resources.reviewers=reviewers(i); q=run_capacity_simulation(c,false); reviewLatency(i)=q.latency.ai_plus_human_review_seconds.mean_seconds/60; end
f=local_fig(); plot(reviewers,reviewLatency,'-o','LineWidth',1.5); grid on; xlabel('Reviewers'); ylabel('Mean reviewed latency (minutes)'); title('Reviewer scaling'); local_save(f,plotDir,'12_reviewers_vs_review_latency.png');

f=local_fig(); demand=[100000 125000 150000 200000]; capacity=repmat(base.sustainability.analytical_capacity_patients_per_year,size(demand)); plot(demand,capacity,'-o',demand,demand,'--','LineWidth',1.5); grid on; xlabel('Annual demand'); ylabel('Annual capacity'); legend('Baseline analytical capacity','Demand','Location','best'); title('Annual demand vs capacity'); local_save(f,plotDir,'13_annual_demand_vs_capacity.png');
f=local_fig(); bar(categorical(scenarioTable.Scenario),scenarioTable.AnnualizedThroughput); grid on; ylabel('Annualized completed cases'); title('Scenario comparison'); local_save(f,plotDir,'14_scenario_comparison.png');
f=local_fig(); bar(categorical(scenarioTable.Scenario),scenarioTable.BottleneckOfferedUtilization); grid on; yline(1,'r--'); ylabel('Bottleneck offered utilization'); title('Bottleneck comparison'); local_save(f,plotDir,'15_bottleneck_comparison.png');
cost=zeros(height(scenarioTable),1); cap=zeros(height(scenarioTable),1); for i=1:height(scenarioTable), c=local_scenario_by_name(scenarioTable.Scenario(i)); cost(i)=c.bandwidth_mbps/1+sum(struct2array(c.resources)); q=results{i}; cap(i)=q.sustainability.analytical_capacity_patients_per_year; end
f=local_fig(); scatter(cost,cap,70,'filled'); grid on; xlabel('Relative resource cost'); ylabel('Analytical annual capacity'); title('Relative resource cost vs capacity'); local_save(f,plotDir,'16_resource_cost_vs_capacity.png');

f=local_fig(); tiledlayout(2,2); nexttile; bar([base.throughput.annualized_completed_patients base.sustainability.analytical_capacity_patients_per_year]); set(gca,'XTickLabel',{'Completed','Capacity'}); ylabel('Patients/year'); title('Scale'); nexttile; bar(base.bottleneck.offered_utilization); ylim([0 1.2]); yline(1,'r--'); title(['Bottleneck: ' base.bottleneck.stage]); nexttile; bar([base.latency.full_report_seconds.mean_seconds base.latency.full_report_seconds.p95_seconds]/60); set(gca,'XTickLabel',{'Mean','P95'}); ylabel('Minutes'); title('Latency'); nexttile; text(0.05,0.8,sprintf('Target: %d/year\nBandwidth: %g Mbps\nVessel workers: %d\nSimulation: %g days',base.workload.annual_patients,base.assumptions.bandwidth_mbps,base.stage.vessel.workers,base.simulation.horizon_days),'FontSize',12); axis off; sgtitle('RETINA-NEXUS Rural Scalability (simulated)'); local_save(f,plotDir,'RETINA_NEXUS_RURAL_SCALABILITY.png');
end

function local_queue_plot(result,plotDir,fieldName,fileName,titleText)
f=local_fig(); q=result.trace.stage_queue_samples.(fieldName); plot(q.time/3600,q.value,'LineWidth',1.2); grid on; xlabel('Simulation time (hours)'); ylabel('Queue length'); title(titleText); local_save(f,plotDir,fileName);
end
function f=local_fig(), f=figure('Visible','off','Color','w'); end
function local_save(f,dir,name), exportgraphics(f,fullfile(dir,name),'Resolution',130); close(f); end
function cfg=local_rederive(cfg)
cfg.required_rate_per_hour=cfg.annual_patients/(cfg.days_per_year*24); cfg.arrival_rate_per_hour=cfg.required_rate_per_hour*cfg.peak_multiplier; cfg.arrival_rate_per_second=cfg.arrival_rate_per_hour/3600; cfg.interarrival_seconds=1/cfg.arrival_rate_per_second; cfg.network_latency_s=cfg.network_latency_ms/1000; cfg.transmission_time_s=(cfg.image_size_mb*8)/cfg.bandwidth_mbps+cfg.network_latency_s;
end
function cfg=local_scenario_by_name(name)
switch char(name)
    case 'BASELINE_RURAL', cfg=scenario_baseline(); case 'POOR_NETWORK', cfg=scenario_poor_network(); case 'PEAK_LOAD', cfg=scenario_peak_load(); case 'COMPUTE_CONSTRAINED', cfg=scenario_compute_constrained(); case 'VESSEL_SCALE_UP', cfg=scenario_vessel_scale(); case 'REVIEW_SCALE_UP', cfg=scenario_review_scale(); otherwise, cfg=scenario_optimized();
end
end
