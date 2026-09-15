function cfg = scenario_baseline()
cfg = retina_nexus_sim_config(struct('scenario_name','BASELINE_RURAL', ...
    'annual_patients',100000,'bandwidth_mbps',10,'peak_multiplier',1.0));
end
