function cfg = scenario_poor_network()
cfg = retina_nexus_sim_config(struct('scenario_name','POOR_NETWORK', ...
    'annual_patients',100000,'bandwidth_mbps',1,'peak_multiplier',1.0));
end
