# SIH Simulation Explanation

## Why Simulink?
Model accuracy alone does not demonstrate whether a rural deployment can process 100,000+ patients/year. This model tests operational scalability.

## What is simulated?
Patient arrival -> image acquisition -> rural bandwidth -> upload queue -> quality gate -> primary AI -> background evidence -> XAI/RetinaGuard -> triage -> ophthalmologist review -> report.

## What is measured?
Throughput, latency, P95/P99 latency, queue growth, worker utilization, review delay, capacity margin, and bottlenecks.

## Architecture boundary
The primary path may complete independently while lesion, R2-V2 vessel, localization, and Grad-CAM/agreement run as bounded/background evidence. Evidence never rewrites the EfficientNet severity output.

## Current bottleneck
The simulation calculates the bottleneck; the baseline result is **vessel_r2v2** at 0.457 offered utilization. Its measured mean service time is 144.5 seconds and maximum is 246.4 seconds.

All outputs are simulated operational estimates. They are not clinical validation, hospital deployment proof, rural network measurements, ophthalmologist performance, or financial cost. Prototype security baseline; production deployment requires authentication/authorization hardening. Neovascularization remains future work / not validated.
