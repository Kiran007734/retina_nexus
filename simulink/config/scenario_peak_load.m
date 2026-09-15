function cfg = scenario_peak_load()
cfg = retina_nexus_sim_config(struct('scenario_name','PEAK_LOAD', ...
    'annual_patients',100000,'bandwidth_mbps',10,'peak_multiplier',3.0));
end
