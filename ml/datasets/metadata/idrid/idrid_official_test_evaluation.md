# IDRiD Official Test Evaluation

Research prototype evaluation only; this is not clinical validation and does not promote the model.

## Dataset and model

- Dataset: IDRiD Disease Grading official Testing Set
- Official test samples: 103
- Model: efficientnet-b0-idrid-20260912-v1
- Checkpoint SHA-256: `d6b8f92bb86a54f98a0295fe8a8b9fadc8938515d03bb5213e6c354344e0b6de`
- Preprocessing: Compose(
    Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True)
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
- Official test images evaluated: 103
- Unprocessable images: 0

## Five-class evaluation

| Metric | Value |
|---|---:|
| Accuracy | 0.631068 |
| Macro Precision | 0.609984 |
| Macro Recall | 0.620342 |
| Macro F1 | 0.581887 |
| QWK | 0.6653704159488063 |
| ROC-AUC OVR macro | 0.8482667013719384 |

Confusion matrix (actual rows, predicted columns):

```text
[[27, 3, 3, 0, 1], [1, 4, 0, 0, 0], [8, 3, 20, 0, 1], [1, 1, 7, 8, 2], [1, 1, 3, 2, 6]]
```

## Referable DR

Referable = grades 2/3/4, with `P2 + P3 + P4 >= 0.5`.

- TP/TN/FP/FN: 50/34/5/14
- Sensitivity: 0.781250
- Specificity: 0.871795
- Precision: 0.909091
- F1: 0.840336
- ROC-AUC: 0.8950320512820513
- SIH target assessment: NOT_MET

## Calibration diagnostics

Raw softmax confidence is not clinically calibrated. No calibration model was fitted on test data. Post-hoc diagnostics only: ECE-10 = `0.109531`, multiclass Brier = `0.538025`, referable Brier = `0.133432`.

## Error analysis

{
  "referable_false_negatives": 14,
  "referable_false_positives": 5,
  "grade_0_to_2_3_4": 4,
  "grade_1_to_2_3_4": 0,
  "grade_2_3_4_to_0_1": 15,
  "grade_3_to_4": 2,
  "grade_4_to_3": 2,
  "high_confidence_incorrect_predictions": 4
}

## Final model status

**KEEP EXPERIMENTAL**

The APTOS production model and production configuration were not changed. The official test result is immutable and must not be used for tuning.
