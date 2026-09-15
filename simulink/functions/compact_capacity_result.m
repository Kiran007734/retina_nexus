function compact = compact_capacity_result(result)
%COMPACT_CAPACITY_RESULT Remove per-patient arrays before JSON/CSV export.
compact = result;
if isfield(compact,'trace'), compact = rmfield(compact,'trace'); end
stageNames = fieldnames(compact.stage);
for i=1:numel(stageNames)
    s=compact.stage.(stageNames{i});
    if isfield(s,'wait_seconds'), s=rmfield(s,'wait_seconds'); end
    if isfield(s,'queue_trace'), s=rmfield(s,'queue_trace'); end
    compact.stage.(stageNames{i})=s;
end
end
