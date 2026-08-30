# Training

scripts/train_classifier.py is the current dataset-driven entry point. It requires a governance split manifest, keeps patient identifiers out of model artifacts, records dataset and model versions, and writes measured metrics without claiming clinical validation. Optional ML packages are listed in backend/requirements-ml.txt.
