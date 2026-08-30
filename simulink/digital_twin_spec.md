# RETINA-NEXUS Simulink Digital Twin specification

## Purpose and boundary

This is an operational digital twin for capacity planning. It estimates
throughput, queues, waiting time, bottlenecks, staff utilization, and resource
requirements under declared scenarios. It does not simulate disease biology,
clinical outcomes, model accuracy, or patient safety and must not be used as
clinical evidence.

The dependency-light reference implementation is
`simulink/retina_nexus_digital_twin.m`. The equivalent Simulink model can be
built from the same parameters and event semantics.

## Top-level blocks

| Block | Inputs | Outputs | Parameters |
| --- | --- | --- | --- |
| Patient Arrival | simulation clock | patient token | patients/day, seed |
| Image Acquisition | patient token | captured image | capture time, acquisition servers |
| Image Quality Gate | captured image | gradable / ungradable | gate time, ungradable rate |
| Recapture Loop | ungradable token | acquisition token or exit | recapture rate, max attempts |
| Secure Transfer | AI result | delivered result | bandwidth delay |
| AI Processing | gradable image | grade/trust token | inference time, AI workers |
| Review Router | grade/trust token | trusted / review / referral | review and referral rates |
| Specialist Queue | review/referral token | reviewed case | specialist count, review time |
| Outcome | reviewed or trusted token | completed outcome | sink counters |

## Signals and units

All times are in minutes unless explicitly labelled. The input parameters are:

- `patients_per_day`
- `image_capture_time`
- `ungradable_rate`
- `recapture_rate`
- `ai_inference_time`
- `bandwidth_delay`
- `specialist_count`
- `review_time`
- `referral_rate`

The reference function also exposes `human_review_rate`, `quality_gate_time`,
`ai_worker_count`, `acquisition_server_count`, `simulation_days`, `seed`, and
`max_recaptures` to make capacity assumptions explicit.

## Decision semantics

Each patient receives one acquisition. An ungradable capture either exits or
returns through the acquisition loop according to `recapture_rate`, capped by
`max_recaptures`. Gradable images enter AI processing. A case enters the
specialist queue when it is referable or sampled for human review. Trusted
non-referable cases can exit without specialist service. Bandwidth delay is
added before downstream result availability.

## Outputs

The reference model returns:

- completed throughput per day;
- maximum and mean specialist queue length;
- mean specialist waiting time;
- acquisition, AI, and specialist utilization;
- stage bottleneck ranking;
- required specialist count at the configured target utilization;
- counts for arrivals, recaptures, ungradable exits, AI cases, reviews, and
  referrals;
- an event table for plotting timelines or importing into Simulink.

The MATLAB output can be connected to `simulink.SimulationData.Dataset` or
logged through To Workspace blocks when the native Simulink model is created.

## Scenarios

The reference scenarios are:

- `NORMAL LOAD`: 60 patients/day, nominal capture and transfer, two
  specialists;
- `HIGH LOAD`: increased arrivals with the same resources;
- `LOW BANDWIDTH`: several minutes of transfer delay;
- `HIGH UNGRADABLE RATE`: poor acquisition conditions and more recapture;
- `LIMITED SPECIALIST CAPACITY`: one specialist with longer queue pressure.

Each scenario is a planning assumption. Replace it with measured site data
before using the results for staffing or deployment decisions.

## Simulink implementation plan

1. Create a discrete-event model with Entity Generator, Queue, Server,
   Entity Transport Delay, and Entity Sink blocks, or equivalent SimEvents
   blocks available in the target MATLAB release.
2. Put the parameters in a `Simulink.Bus` named `RetinaNexusConfig` and load
   scenario values from a MATLAB script.
3. Use a Stateflow chart for the quality/recapture decision and a second chart
   for trusted, human-review, referral, and specialist routing.
4. Log queue length, entity wait time, server busy time, and sink counts with
   named signals matching the reference function outputs.
5. Run each scenario with multiple seeds, compare means and percentile ranges,
   and document warm-up/measurement windows.
6. Calibrate arrival, capture, ungradable, transfer, and review distributions
   from de-identified operational data. Do not tune parameters to clinical
   outcomes without an approved study design.
