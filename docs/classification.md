# DR classification pipeline

RETINA-NEXUS includes a reproducible, configurable diabetic-retinopathy classification pipeline. It is an engineering implementation boundary and does not claim clinical validation, regulatory clearance, or target metrics.

## Severity classes

| Level | Label |
| --- | --- |
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

The default referable mapping is moderate_or_worse: grades 2, 3, and 4 are referable. This is a configurable engineering choice, not a universal clinical rule. Set REFERABLE_MIN_GRADE for inference or use --referable-min-grade for training and benchmarking. The mapping is stored in checkpoints and returned by the API.

## Model strategy

The model factory exposes EfficientNet B0/B1/B2, ResNet 18/50, and MobileNet V3 Small as the lightweight baseline. All use a shared feature extractor with heads for DR versus No DR, Referable versus Non-Referable, and five-level severity grading.

Fine-grade training supports normal five-class softmax and an experimental cumulative ordinal head. Ordinal mode is not presented as a validated clinical method.

With --pretrained, torchvision official weight mechanisms are used. If weights cannot be imported or downloaded, training stops with an actionable error. No weights or metrics are fabricated. Use --no-pretrained only for an explicitly uninitialized baseline.

## Training and benchmarking

Install optional dependencies with:

    pip install -r backend/requirements-ml.txt

After authorized data placement and split generation, run:

    python scripts/validate_dataset.py --dataset aptos2019
    python scripts/create_splits.py --dataset aptos2019
    python scripts/train_classifier.py --dataset aptos2019 --dataset-version aptos2019-authorized-v1 --backbone efficientnet_b0

Training supports batch size, learning rate, epochs, early stopping, mixed precision, reproducible seeds, checkpointing, weighted loss, focal loss, and weighted sampling. The split manifest is checked for cross-split patient and duplicate groups before training.

The training artifact contains best and last checkpoints, training configuration, history, measured validation/test metrics, and a SHA-256 checksum. A local model registry index is written under ml/weights/model_registry.json. Weights and generated registries are ignored by Git.

Run scripts/benchmark_classifiers.py with repeated NAME=PATH checkpoint arguments to compare EfficientNet, ResNet, and MobileNet candidates on the same manifest and split. The report includes accuracy, sensitivity, specificity, precision, recall, F1, ROC-AUC where computable, confusion matrix, and referable DR sensitivity/specificity. Ranking is for reporting only and does not declare a universally superior model.

## API and inference

Set CLASSIFIER_MODEL_PATH to a trained checkpoint and restart the backend. POST /api/v1/screening/classify accepts an image_id and returns the grade and label, five probabilities, referable result and probability, raw confidence, model version, backbone, hierarchical probabilities, and referable mapping.

Raw confidence is not calibrated confidence and is not a clinical trust guarantee. An absent, incompatible, or unimportable model returns HTTP 503 with setup instructions. The API never returns a fabricated prediction.
