# NORMALIZATION_CONTRACT.md — norm-v1.2.0 (FROZEN)

> **Status**: FROZEN — do not change field names, types, or semantics without
> bumping the version in `normalize.py` and updating this document.
>
> **Last updated**: 2026-09-27
> **Module**: [`normalize.py`](file:///c:/Users/Asus/Desktop/AmazonML/normalize.py)
> **Version**: `norm-v1.2.0`

---

## Purpose

This document defines the **single source of truth** for how raw S1/S2/S3
business-entity records are normalized for the Amazon ML Challenge 2026
(Business Entity Resolution).

**All downstream consumers** — blocking (Nishant/Piyush), feature engineering
(Parimarjan), and the matching model — **must import from `normalize.py`**
rather than re-implementing normalization independently.

---

## Core 15-Field Contract

Every call to `normalize_record(entity_id, business_name, business_address, country)`
returns a `NormalizedRecord` dataclass with at minimum these 15 fields:

| # | Field | Type | Source | Description |
|---|-------|------|--------|-------------|
| 1 | `entity_id` | `str` | pass-through | Original entity ID (e.g. `S1-00001`) |
| 2 | `source` | `str` | derived | `"S1"` / `"S2"` / `"S3"` from entity_id prefix |
| 3 | `country_raw` | `str \| None` | pass-through | Untouched country string |
| 4 | `country_norm` | `str \| None` | computed | Lowercased, whitespace-collapsed, open-set |
| 5 | `name_raw` | `str \| None` | pass-through | Untouched business name |
| 6 | `name_norm` | `str \| None` | computed | NFKC → lowercase → `&`→`and` → punct-stripped → legal-form-mapped. **Preserves accents.** |
| 7 | `name_core` | `str \| None` | computed | `name_norm` with legal-form tokens removed (brand identity only) |
| 8 | `name_tokens` | `List[str]` | computed | Order-preserved tokens from `name_norm` |
| 9 | `address_raw` | `str \| None` | pass-through | Untouched business address |
| 10 | `address_norm` | `str \| None` | computed | NFKC → lowercase → `&`→`and` → punct-stripped → abbreviations expanded |
| 11 | `address_tokens` | `List[str]` | computed | Order-preserved tokens from `address_norm` |
| 12 | `numeric_tokens` | `List[str]` | computed | Digit-bearing tokens extracted from address (house/building numbers) |
| 13 | `postal_tokens` | `List[str]` | computed | Postal/PIN/ZIP-like tokens (5-6 digit patterns) |
| 14 | `script_flags` | `Dict` | computed | `{name: {scripts_present, primary_script, is_mixed_script}, address: {...}}` |
| 15 | `missingness_flags` | `Dict[str, bool]` | computed | `{name_missing, address_missing, country_missing}` |

---

## Additive Fields (beyond core 15)

These fields exist in the `NormalizedRecord` but are **not part of the frozen
core contract** — they may be added/modified in future versions without
breaking the core 15:

| Field | Type | Description |
|-------|------|-------------|
| `name_fold` | `str \| None` | Accent-folded `name_norm` (é→e). For blocking/comparison. |
| `address_fold` | `str \| None` | Accent-folded `address_norm`. For blocking/comparison. |
| `name_had_dba_marker` | `bool` | HYPOTHESIS: DBA/AKA/trading-as pattern detected |
| `name_had_leading_code_artifact` | `bool` | HYPOTHESIS: Leading record-code pattern detected |
| `name_token_counts` | `Dict[str, int]` | Token frequency within the name |
| `normalization_version` | `str` | Version tag for traceability |

---

## Design Rules (invariants)

1. **Raw values are never overwritten.** `*_raw` fields always carry the
   untouched original string, regardless of normalization.

2. **Missingness is explicit.** `None` means blank/missing. Never literal
   `"nan"` / `"none"` / `""`. The `missingness_flags` dict provides boolean
   convenience accessors.

3. **Country is open-set.** No hard-coded allow-list of `{US, India}`.
   France, Wakanda, or any other string passes through `normalize_country()`
   untouched (case/whitespace normalized only).

4. **Legal-form mapping is configurable.** `LEGAL_FORM_MAP` in `normalize.py`
   controls which suffixes are canonicalized. `name_core` strips them;
   `name_norm` / `name_tokens` retain them.

5. **No blind transliteration.** Devanagari, Arabic, CJK, Cyrillic text is
   flagged via `script_flags` and preserved as-is. The accent-folding in
   `name_fold` / `address_fold` is restricted to Latin diacritics (U+0300–U+036F)
   and does not corrupt non-Latin scripts.

6. **Determinism.** Same input + same config → byte-identical output. Verified
   by `test_normalize.py::TestDeterminism`.

7. **Zero third-party dependencies.** `normalize.py` uses stdlib only.
   The optional `normalize_dataframe()` helper does a deferred `import pandas`.

---

## Consumer Integration Guide

### Blocking (Nishant / `blocking.py`)

```python
from normalize import (
    normalize_name, normalize_address, normalize_country,
    LEGAL_FORM_TOKENS, NORMALIZATION_VERSION,
)

# Use name_fold (accent-stripped) for blocking keys
out = normalize_name("Café du Commerce")
blocking_key = out["name_fold"]  # "cafe du commerce"

# Use address_fold for address blocking keys  
out = normalize_address("Rue de l'Église, Paris")
addr_key = out["address_fold"]  # accent-safe
```

### Feature Engineering (Parimarjan)

```python
from normalize import normalize_record

rec = normalize_record("S1-00001", "Sharma Textiles Pvt Ltd", "12 MG Road, 560001", "India")
# Access any field:
rec.name_norm      # "sharma textiles pvt ltd"
rec.name_core      # "sharma textiles" (legal forms removed)
rec.name_tokens    # ["sharma", "textiles", "pvt", "ltd"]
rec.postal_tokens  # ["560001"]
rec.script_flags   # {"name": {"primary_script": "latin", ...}, ...}
```

### Batch Processing

```python
from normalize import normalize_dataframe
import pandas as pd

df = pd.read_csv("train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
norm_df = normalize_dataframe(df)  # returns flat DataFrame with all fields
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| `norm-v1.0.0` | 2026-09-26 | Initial: 15-field core contract |
| `norm-v1.1.0` | 2026-09-26 | Added `name_fold`, `address_fold`, DBA/artifact marker flags |
| `norm-v1.2.0` | 2026-09-27 | Added `normalize_dataframe()`, froze core-15 contract |

---

## Prohibited

- **No external APIs, geocoding, or business databases.** All logic is pure
  string processing over the provided fields, per challenge rules.
- **No downstream re-normalization.** If blocking or features need a different
  text representation, add it as an additive field in `normalize.py` and bump
  the version — never re-implement independently.
