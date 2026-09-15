# GitHub Release Manifest

This classification is packaging guidance only; it does not grant dataset or model redistribution rights.

| Artifact class | Classification | Decision basis |
|---|---|---|
| Source code, configs, tests, and documentation | **PUSH** | Keep source-controlled. |
| Raw APTOS/IDRiD/DRIVE/Messidor datasets, labels, masks, and archives | **DO NOT PUSH / EXTERNAL REQUIRED** | Keep local/external and require documented authorization. |
| Dataset metadata and evaluation reports | **PUSH WITH REVIEW** | Retain only after provenance and sensitivity review. |
| APTOS production classifier checkpoint | **EXTERNAL REQUIRED** | Keep local/external and require documented authorization. |
| Optional lesion and R2-V2 vessel checkpoints | **EXTERNAL REQUIRED; Git LFS only after license review** | Keep local/external and require documented authorization. |
| IDRiD/DRIVE research checkpoints and cross-validation folds | **OPTIONAL / EXTERNAL REQUIRED** | Keep local/external and require documented authorization. |
| RETGUARD ONNX verifier and OOD artifact | **OPTIONAL RESEARCH / EXTERNAL REQUIRED** | Keep local/external and require documented authorization. |
| Simulink .slx, source scripts, plots, and compact results | **PUSH** | Keep source-controlled. |
| Secrets, .env files, uploads, databases, node_modules, caches, and logs | **DO NOT PUSH** | Keep local/external and require documented authorization. |
