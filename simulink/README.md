# Simulink integration boundary

The Simulink boundary is a systems-engineering digital twin, not a clinical
model and not a replacement for the screening API. The model is intended to
stress the operational flow from patient arrival through acquisition,
recapture, AI processing, specialist review, and outcome.

- [digital_twin_spec.md](digital_twin_spec.md) defines the blocks, signals,
  scenarios, assumptions, and Simulink implementation mapping.
- [retina_nexus_digital_twin.m](retina_nexus_digital_twin.m) is a
  dependency-light MATLAB-compatible discrete-event prototype. It can be used
  to calibrate scenarios before building the equivalent Simulink model.
- [run_digital_twin_scenarios.m](run_digital_twin_scenarios.m) runs the five
  declared load/bandwidth/capacity scenarios and returns a comparison table.

The acquisition adapter must emit the same validated image object used by
manual uploads. It must not bypass the Image Trust Gate, audit logging, model
version checks, or human review.
