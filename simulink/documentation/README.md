# RETINA-NEXUS Simulink Digital Twin

This directory contains a real MATLAB R2026a/Simulink/SimEvents system-level deployment model. Run `build_model`, `run_baseline`, `run_all_scenarios`, `run_parameter_sweep`, `optimize_resources`, `validate_model`, `validate_reproducibility`, and `generate_final_report` from MATLAB with `simulink` on the path.

The model uses measured project inference timings as service parameters and does not execute neural networks for simulated patients. All workload, network, image-size, rejection, staffing, and review parameters are explicitly labeled assumptions.

Model: `RETINA_NEXUS_SYSTEM.slx`. Results are under `results/`; plots are under `plots/`; traceability is under `documentation/`.
