# RETINA-NEXUS Simulation Parameter Provenance

This is an operational capacity simulation, not clinical evidence.

| Parameter | Value | Source | Type |
|---|---:|---|---|
| Annual workload | 100,000 patients/year | SIH problem statement | SIH requirement |
| Workload targets | 100k, 125k, 150k, 200k/year | SIH problem statement | SIH requirement |
| Classification service time | 0.209 s mean / 0.275 s max | pre-Simulink runtime artifact | Measured project runtime |
| Lesion service time | 6.28 s mean / 10.62 s max | pre-Simulink runtime artifact | Measured project runtime |
| Localization service time | 0.0595 s mean / 0.099 s max | pre-Simulink runtime artifact | Measured project runtime |
| R2-V2 vessel service time | 144.5 s mean / 246.4 s max | pre-Simulink runtime artifact | Measured project runtime |
| Grad-CAM/agreement service time | 10.2 s mean / 15.7 s max | pre-Simulink runtime artifact | Measured project runtime |
| RetinaGuard service time | 0.0088 s mean / 0.0171 s max | pre-Simulink runtime artifact | Measured project runtime |
| PDF service time | 0.00186 s mean / 0.00355 s max | pre-Simulink runtime artifact | Measured project runtime |
| Image payload | 5 MB | Configured default | Simulation assumption |
| Baseline bandwidth | 10 Mbps | Configured rural scenario | Simulation assumption |
| Quality rejection | 8.0% | Configured default | Simulation assumption |
| Review time | 2 minutes | Configured default | Simulation assumption |
| Hourly demand | 11.415525 patients/hour at 100k/year | annual_patients/(365*24) | Derived value |
| Transmission delay | (image MB*8)/Mbps + latency | network model | Derived value |

Network rates, image size, quality/review probabilities, staffing, and timing mode are replaceable assumptions. They are not rural-site measurements or clinical claims.
