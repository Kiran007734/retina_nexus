# Simulink architecture

Acquisition adapters should emit validated image objects into the same storage
and quality-gate interfaces used by manual uploads. Device integration must not
bypass quality assessment, audit logging, or human review.

The proposed top-level model is:

```text
Patient Arrival -> Image Acquisition -> Image Quality Gate
                         ^                    |
                         |------ Recapture <-+
                              |
                              v
                       AI Processing
                              |
             Trusted / Human Review / Referral
                              |
                       Specialist Queue
                              |
                            Outcome
```

Use the MATLAB function in this folder as the reference behavior for a
Simulink implementation with discrete-event queues and configurable source,
server, transport, and sink blocks.
