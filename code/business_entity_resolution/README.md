# Business Entity Resolution Pipeline
### Amazon ML Challenge 2026

## 1. Structure
```text
code/business_entity_resolution/
├── src/
│   ├── normalize.py      # Entity normalization module (norm-v1.2.0 contract)
│   ├── blocking.py       # Multi-pass candidate blocking (Passes A-E, bounded K=30)
│   ├── features.py       # 30-dim frozen pair feature extractor
│   ├── model.py          # HistGradientBoostingClassifier candidate matcher
│   ├── decision.py       # Threshold-based set decision layer & conflict suppression
│   ├── evaluate.py       # Entity-level Macro F0.5 evaluation engine
│   └── pipeline.py       # End-to-end streaming inference & submission file writer
├── tests/                # 150/150 passing unit test suite
├── requirements.txt      # Python dependencies
└── README.md             # Pipeline architecture & reproduction guide
```

## 2. Reproduction Instructions

To reproduce the submission files `output/matching_results.tsv` and `output/candidate_pairs.tsv` on the test dataset:

```powershell
# Set PYTHONPATH
$env:PYTHONPATH="code\business_entity_resolution\src"

# Run end-to-end streaming inference
python code/business_entity_resolution/src/pipeline.py `
  --train-dir dataset/train `
  --test-dir dataset/test `
  --output-dir output
```

## 3. Validation

Run the official submission validator:

```powershell
python utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir dataset/test `
  --check-ids
```
