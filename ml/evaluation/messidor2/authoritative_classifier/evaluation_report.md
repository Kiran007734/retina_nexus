# Messidor-2 authoritative-original classifier evaluation

Status: **DESCRIPTIVE_EXTERNAL_EVALUATION**

This run uses the frozen APTOS EfficientNet-B0 checkpoint against the authoritative extracted original images. It is not clinical validation and no threshold was tuned on Messidor-2.

- Original evaluation records: `1744`
- Successful inferences: `1744`
- Checkpoint: `ml/weights/classifiers/aptos2019/efficientnet-b0-aptos2019-20260830-v1/checkpoint_best.pt`
- Checkpoint SHA-256: `ae6bb62ced2a108abc1a862870e64985b368b84e69bd8c8c8aa9912754d1a70b`
- Checkpoint unchanged: `True`
- Label source: `ml/datasets/raw/messidor/images/messidor_data.csv`
- Label provenance warning: Labels are local/adjudicated reference labels and not independently proven official Messidor-2 ground truth.

## Severity metrics

```json
{
  "accuracy": 0.6032110091743119,
  "averaging": {
    "macro_f1": "unweighted mean across five classes",
    "roc_auc_ovr_macro": "one-vs-rest macro average",
    "weighted_f1": "support-weighted mean across five classes"
  },
  "class_distribution": {
    "0": 1017,
    "1": 270,
    "2": 347,
    "3": 75,
    "4": 35
  },
  "clinical_validation_claim": false,
  "confusion_matrix": [
    [
      932,
      1,
      52,
      19,
      13
    ],
    [
      247,
      2,
      15,
      4,
      2
    ],
    [
      217,
      4,
      80,
      33,
      13
    ],
    [
      5,
      0,
      34,
      28,
      8
    ],
    [
      2,
      0,
      9,
      14,
      10
    ]
  ],
  "evaluation_population": "Successful inference on readable, label-matched, gradable authoritative original images",
  "exclusions": "Unmatched label rows are excluded; ungradable rows and unsuccessful inferences are excluded from grade metrics.",
  "label_mapping": "Messidor-2 adjudicated five-point ICDR grade 0..4 mapped 1:1 to the existing APTOS output labels defined in the downloaded label readme.",
  "macro_f1": 0.3306505904354619,
  "precision_macro": 0.3748326625544617,
  "prediction_distribution": {
    "0": 1403,
    "1": 7,
    "2": 190,
    "3": 98,
    "4": 46
  },
  "quadratic_weighted_kappa": 0.4923817586206797,
  "recall_macro": 0.3626846845023377,
  "referable_dr_grade_2_or_worse": {
    "f1": 0.5981735159817351,
    "false_negative": 195,
    "false_positive": 157,
    "precision": 0.6252983293556086,
    "recall": 0.5733041575492341,
    "roc_auc": 0.8185252627265757,
    "rule": "true and predicted referable if grade >= 2",
    "sensitivity": 0.5733041575492341,
    "specificity": 0.878010878010878,
    "true_negative": 1130,
    "true_positive": 262
  },
  "roc_auc_ovr_macro": 0.7689673671300099,
  "sample_count": 1744,
  "status": "CALCULATED",
  "weighted_f1": 0.5295583124650238
}
```

## Referable threshold comparison

```json
{
  "threshold_0.20": {
    "f1": 0.5659301496792587,
    "false_negative": 60,
    "false_positive": 549,
    "pr_auc": 0.6750497124694614,
    "precision": 0.41966173361522197,
    "recall": 0.8687089715536105,
    "roc_auc": 0.8185252627265757,
    "rule": "referable_probability = P(2)+P(3)+P(4); referable = referable_probability >= threshold; severity grade = argmax(P0..P4).",
    "sample_count": 1744,
    "sensitivity": 0.8687089715536105,
    "specificity": 0.5734265734265734,
    "threshold": 0.2,
    "true_negative": 738,
    "true_positive": 397
  },
  "threshold_0.50": {
    "f1": 0.5981735159817351,
    "false_negative": 195,
    "false_positive": 157,
    "pr_auc": 0.6750497124694614,
    "precision": 0.6252983293556086,
    "recall": 0.5733041575492341,
    "roc_auc": 0.8185252627265757,
    "rule": "referable_probability = P(2)+P(3)+P(4); referable = referable_probability >= threshold; severity grade = argmax(P0..P4).",
    "sample_count": 1744,
    "sensitivity": 0.5733041575492341,
    "specificity": 0.878010878010878,
    "threshold": 0.5,
    "true_negative": 1130,
    "true_positive": 262
  }
}
```

Threshold 0.20 is an IDRiD-development research threshold. Threshold 0.50 is the established grade-2-or-worse rule. Neither was selected using Messidor-2 results.

The four archive originals without local labels are excluded by the authoritative manifest and are not present in this evaluation.
