function cfg = scenario_optimized()
% The optimizer writes a separate configuration; this is a safe baseline
% placeholder for running the named scenario before optimization completes.
cfg = retina_nexus_sim_config(struct('scenario_name','OPTIMIZED', ...
    'annual_patients',100000,'bandwidth_mbps',25,'peak_multiplier',1.0));
end
