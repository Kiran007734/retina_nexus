function result = run_capacity_simulation(cfg, includeTrace)
%RUN_CAPACITY_SIMULATION Reproducible operational discrete-event runner.
% It uses measured service times and does not execute neural networks per
% entity. This is not a clinical simulation.
if nargin < 2, includeTrace = false; end
rng(cfg.random_seed, 'twister');
horizon = cfg.simulation_seconds;
arrivals = local_arrivals(cfg.arrival_rate_per_second, horizon, cfg.arrival_process);
n = numel(arrivals);
s = cfg.active_service_times; r = cfg.resources;

captureStart=NaN(n,1); captureEnd=NaN(n,1); uploadStart=NaN(n,1); uploadEnd=NaN(n,1); qualityEnd=NaN(n,1);
primaryStart=NaN(n,1); primaryEnd=NaN(n,1); retinaEnd=NaN(n,1);
reviewStart=NaN(n,1); reviewEnd=NaN(n,1); reportStart=NaN(n,1); reportEnd=NaN(n,1);
lesionStart=NaN(n,1); lesionEnd=NaN(n,1); vesselStart=NaN(n,1); vesselEnd=NaN(n,1);
localizationStart=NaN(n,1); localizationEnd=NaN(n,1); xaiStart=NaN(n,1); xaiEnd=NaN(n,1);
attempts=zeros(n,1); recaptures=zeros(n,1); referrals=false(n,1); reviewRequired=false(n,1); ungradableExit=false(n,1);

captureAvail=zeros(1,r.acquisition_workers); uploadAvail=zeros(1,r.upload_workers); qualityAvail=zeros(1,r.acquisition_workers);
primaryAvail=zeros(1,r.classifier_workers); retinaAvail=zeros(1,r.classifier_workers); reviewAvail=zeros(1,r.reviewers);
reportAvail=zeros(1,r.report_workers); lesionAvail=zeros(1,r.lesion_workers); vesselAvail=zeros(1,r.vessel_workers);
localizationAvail=zeros(1,r.localization_workers); xaiAvail=zeros(1,r.xai_workers);
ungradableCaptures=0; ungradableExits=0; recaptureCount=0;

for i=1:n
    ready=arrivals(i); captured=false;
    while ~captured
        attempts(i)=attempts(i)+1;
        [captureStart(i),captureEnd(i),captureAvail]=local_schedule(ready,captureAvail,s.capture);
        [uploadStart(i),uploadEnd(i),uploadAvail]=local_schedule(captureEnd(i),uploadAvail,cfg.transmission_time_s);
        [~,qualityEnd(i),qualityAvail]=local_schedule(uploadEnd(i),qualityAvail,s.quality_gate);
        if rand() >= cfg.quality_rejection_rate
            captured=true;
        else
            ungradableCaptures=ungradableCaptures+1;
            if attempts(i)<=cfg.max_recaptures && rand()<cfg.recapture_probability
                recaptures(i)=recaptures(i)+1; recaptureCount=recaptureCount+1; ready=qualityEnd(i);
            else
                ungradableExit(i)=true; ungradableExits=ungradableExits+1; break;
            end
        end
    end
    if ~captured, continue; end
    [primaryStart(i),primaryEnd(i),primaryAvail]=local_schedule(qualityEnd(i),primaryAvail,s.classification);
    [~,retinaEnd(i),retinaAvail]=local_schedule(primaryEnd(i),retinaAvail,s.retina_guard);
    referrals(i)=rand()<cfg.referral_probability;
    reviewRequired(i)=referrals(i) || rand()<cfg.review_probability;
    if reviewRequired(i)
        [reviewStart(i),reviewEnd(i),reviewAvail]=local_schedule(retinaEnd(i),reviewAvail,cfg.review_time_minutes*60);
    end
    reportStart(i)=retinaEnd(i);
    if isfinite(reviewEnd(i)), reportStart(i)=max(reportStart(i),reviewEnd(i)); end
    [reportStart(i),reportEnd(i),reportAvail]=local_schedule(reportStart(i),reportAvail,s.pdf);

    % Evidence is a bounded/background branch and never rewrites severity.
    [lesionStart(i),lesionEnd(i),lesionAvail]=local_schedule(primaryEnd(i),lesionAvail,s.lesion);
    [vesselStart(i),vesselEnd(i),vesselAvail]=local_schedule(lesionEnd(i),vesselAvail,s.vessel);
    [localizationStart(i),localizationEnd(i),localizationAvail]=local_schedule(vesselEnd(i),localizationAvail,s.localization);
    [xaiStart(i),xaiEnd(i),xaiAvail]=local_schedule(localizationEnd(i),xaiAvail,s.xai_agreement);
end

gradable=isfinite(primaryEnd); primaryCompleted=gradable & primaryEnd<=horizon;
reportCompleted=gradable & reportEnd<=horizon; evidenceCompleted=gradable & xaiEnd<=horizon;
ungradableCompleted=ungradableExit & qualityEnd<=horizon; outcomeCompleted=primaryCompleted|ungradableCompleted;

stage=struct();
stage.upload=local_stage(uploadStart,uploadEnd,uploadAvail,r.upload_workers,cfg.transmission_time_s,horizon,'upload');
stage.quality=local_stage(uploadEnd,qualityEnd,qualityAvail,r.acquisition_workers,s.quality_gate,horizon,'quality_gate');
stage.classifier=local_stage(qualityEnd(gradable),primaryEnd(gradable),primaryAvail,r.classifier_workers,s.classification,horizon,'classifier');
stage.retina_guard=local_stage(primaryEnd(gradable),retinaEnd(gradable),retinaAvail,r.classifier_workers,s.retina_guard,horizon,'retina_guard');
stage.lesion=local_stage(primaryEnd(gradable),lesionEnd(gradable),lesionAvail,r.lesion_workers,s.lesion,horizon,'lesion');
stage.vessel=local_stage(lesionEnd(gradable),vesselEnd(gradable),vesselAvail,r.vessel_workers,s.vessel,horizon,'vessel_r2v2');
stage.localization=local_stage(vesselEnd(gradable),localizationEnd(gradable),localizationAvail,r.localization_workers,s.localization,horizon,'localization');
stage.xai=local_stage(localizationEnd(gradable),xaiEnd(gradable),xaiAvail,r.xai_workers,s.xai_agreement,horizon,'xai_agreement');
reviewMask=isfinite(reviewStart);
stage.review=local_stage(retinaEnd(reviewMask),reviewEnd(reviewMask),reviewAvail,r.reviewers,cfg.review_time_minutes*60,horizon,'human_review');
stage.reporting=local_stage(reportStart(gradable),reportEnd(gradable),reportAvail,r.report_workers,s.pdf,horizon,'reporting');

stageNames=fieldnames(stage); offered=zeros(numel(stageNames),1); margins=zeros(numel(stageNames),1);
for k=1:numel(stageNames), offered(k)=stage.(stageNames{k}).offered_utilization; margins(k)=stage.(stageNames{k}).capacity_margin; end
[~,ix]=max(offered); bottleneck=stage.(stageNames{ix});
aiOnly=retinaEnd(gradable)-arrivals(gradable); primary=primaryEnd(gradable)-arrivals(gradable);
reviewLat=reportEnd(reviewRequired&gradable)-arrivals(reviewRequired&gradable);
full=reportEnd(gradable)-arrivals(gradable); evidence=xaiEnd(gradable)-arrivals(gradable);

result=struct(); result.schema_version='retina-nexus-capacity-result-v1'; result.scenario=cfg.scenario_name;
result.simulation=struct('horizon_days',cfg.simulation_days,'horizon_seconds',horizon,'seed',cfg.random_seed,'arrival_process',cfg.arrival_process);
result.workload=struct('annual_patients',cfg.annual_patients,'peak_multiplier',cfg.peak_multiplier,'generated',n, ...
    'required_patients_per_hour',cfg.required_rate_per_hour,'arrival_patients_per_hour',cfg.arrival_rate_per_hour);
result.counts=struct('generated',n,'gradable',sum(gradable),'primary_completed',sum(primaryCompleted), ...
    'report_completed',sum(reportCompleted),'evidence_completed',sum(evidenceCompleted), ...
    'ungradable_captures',ungradableCaptures,'ungradable_exits',ungradableExits,'recaptures',recaptureCount, ...
    'review_required',sum(reviewRequired),'reviews_completed',sum(reviewRequired&reviewEnd<=horizon), ...
    'referrals',sum(referrals),'patient_outcomes_completed',sum(outcomeCompleted));
result.pending=struct('primary',sum(gradable&~primaryCompleted),'report',sum(gradable&~reportCompleted), ...
    'evidence',sum(gradable&~evidenceCompleted),'review',sum(reviewRequired&~(reviewEnd<=horizon)));
result.stage=stage;
result.bottleneck=struct('stage',bottleneck.name,'utilization',bottleneck.utilization, ...
    'offered_utilization',bottleneck.offered_utilization,'capacity_margin',bottleneck.capacity_margin, ...
    'reason','Highest offered utilization among modeled service stations');
result.latency=struct('ai_only_seconds',local_latency(aiOnly),'primary_seconds',local_latency(primary), ...
    'ai_plus_human_review_seconds',local_latency(reviewLat),'full_report_seconds',local_latency(full), ...
    'background_evidence_seconds',local_latency(evidence));
maxRate=local_max_rate(cfg,stage);
result.throughput=struct('completed_patients_per_second',sum(outcomeCompleted)/horizon, ...
    'completed_patients_per_minute',sum(outcomeCompleted)/horizon*60, ...
    'completed_patients_per_hour',sum(outcomeCompleted)/horizon*3600, ...
    'completed_patients_per_day',sum(outcomeCompleted)/max(cfg.simulation_days,eps), ...
    'annualized_completed_patients',sum(outcomeCompleted)/max(cfg.simulation_days,eps)*cfg.days_per_year, ...
    'maximum_sustainable_patients_per_hour',maxRate);
pendingValues=struct2array(result.pending);
primaryPathSustainable=(result.pending.primary==0 && result.pending.report==0 && result.pending.review==0);
result.sustainability=struct('annual_target',cfg.annual_patients,'analytical_capacity_patients_per_year',maxRate*cfg.days_per_year*24, ...
    'all_queues_stable',all(margins>=1),'simulation_pending_zero',all(pendingValues==0), ...
    'primary_path_sustainable',primaryPathSustainable, ...
    'background_evidence_pending',result.pending.evidence, ...
    'sustainable',all(margins>=1)&&primaryPathSustainable, ...
    'criterion','Stable modeled service capacity and no pending primary/review outcome at the simulation horizon; background evidence tail reported separately');
result.assumptions=struct('quality_rejection_rate',cfg.quality_rejection_rate,'recapture_probability',cfg.recapture_probability, ...
    'review_probability',cfg.review_probability,'referral_probability',cfg.referral_probability,'image_size_mb',cfg.image_size_mb, ...
    'bandwidth_mbps',cfg.bandwidth_mbps,'network_latency_ms',cfg.network_latency_ms,'timing_mode',cfg.timing_mode, ...
    'note','Simulation assumptions are not clinical/site measurements');
result.integrity=struct('clinical_predictions_generated',false,'model_weights_changed',false,'thresholds_changed',false,'datasets_acquired',false);
if includeTrace
    result.trace=struct('arrivals',arrivals,'primary_latency',primary,'full_latency',full,'evidence_latency',evidence, ...
        'vessel_wait',stage.vessel.wait_seconds,'review_wait',stage.review.wait_seconds, ...
        'stage_queue_samples',structfun(@(z)z.queue_trace,stage,'UniformOutput',false));
end
end

function arrivals=local_arrivals(lambda,horizon,processName)
if lambda<=0, arrivals=zeros(0,1); return; end
values=zeros(max(100,ceil(lambda*horizon*1.25)),1); count=0; clock=0;
while clock<horizon
    if strcmpi(processName,'POISSON'), clock=clock-log(max(rand(),realmin))/lambda; else, clock=clock+1/lambda; end
    if clock>=horizon, break; end
    count=count+1; if count>numel(values), values=[values;zeros(numel(values),1)]; end %#ok<AGROW>
    values(count)=clock;
end
arrivals=values(1:count);
end

function [startTime,endTime,available]=local_schedule(release,available,duration)
[~,worker]=min(available); startTime=max(release,available(worker)); endTime=startTime+duration; available(worker)=endTime;
end

function out=local_stage(release,finish,available,workers,serviceTime,horizon,name)
release=release(:); finish=finish(:); valid=isfinite(release)&isfinite(finish); release=release(valid); finish=finish(valid);
start=finish-serviceTime; jobs=numel(release); wait=max(start-release,0); offered=jobs*serviceTime/max(horizon*workers,eps);
capacityPerHour=workers*3600/max(serviceTime,eps); demandPerHour=jobs/max(horizon,eps)*3600; margin=capacityPerHour/max(demandPerHour,eps);
[qt,qv,mq,meanq]=local_queue(release,start,horizon);
out=struct('name',name,'jobs',jobs,'completed_by_horizon',sum(finish<=horizon),'workers',workers,'service_seconds',serviceTime, ...
    'capacity_per_hour',capacityPerHour,'demand_per_hour',demandPerHour,'capacity_margin',margin, ...
    'offered_utilization',offered,'utilization',min(1,offered),'mean_wait_seconds',local_mean(wait), ...
    'p95_wait_seconds',local_percentile(wait,95),'max_wait_seconds',local_max(wait),'max_queue',mq,'mean_queue',meanq, ...
    'stable',margin>=1,'wait_seconds',wait,'queue_trace',struct('time',qt,'value',qv));
end

function [times,values,maxQueue,meanQueue]=local_queue(release,start,horizon)
if isempty(release), times=[0;horizon]; values=[0;0]; maxQueue=0; meanQueue=0; return; end
events=sortrows([release ones(numel(release),1);start -ones(numel(start),1)],[1 2]);
times=[0;events(:,1);horizon]; values=zeros(size(times)); q=0;
for i=2:numel(times), same=events(:,1)==times(i); if any(same), q=q+sum(events(same,2)); end; values(i)=max(q,0); end
maxQueue=max(values); meanQueue=sum(values(1:end-1).*diff(times))/max(horizon,eps);
end

function out=local_latency(values)
values=values(isfinite(values)); out=struct('count',numel(values),'mean_seconds',local_mean(values),'median_seconds',local_percentile(values,50), ...
    'p90_seconds',local_percentile(values,90),'p95_seconds',local_percentile(values,95),'p99_seconds',local_percentile(values,99),'max_seconds',local_max(values));
end
function value=local_percentile(values,p)
if isempty(values), value=0; else, values=sort(values(:)); value=values(max(1,min(numel(values),ceil(p/100*numel(values))))); end
end
function value=local_mean(values), if isempty(values), value=0; else, value=mean(values); end, end
function value=local_max(values), if isempty(values), value=0; else, value=max(values); end, end
function value=local_max_rate(cfg,stage)
names=fieldnames(stage); rates=zeros(numel(names),1); for i=1:numel(names), rates(i)=stage.(names{i}).capacity_per_hour; end
value=min(rates)/max(1-cfg.quality_rejection_rate,eps);
end
