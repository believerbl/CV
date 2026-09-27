"""
features.py
===========
Pair-feature engineering module for Amazon ML Challenge 2026.
Owned by Parimarjan Shukla (+ Nidhi normalization contract).

Consumes NormalizedRecord instances produced by `normalize.py` (norm-v1.2.0)
and computes a fixed, documented, and deterministic feature vector for every
candidate pair (S1, S2/S3).

Feature Families:
  1. Name lexical & structural similarities (Jaccard, edit distance, exact core)
  2. Address lexical & structural similarities (token Jaccard, edit distance)
  3. Numeric address evidence (perfect, partial, none, conflict)
  4. Postal code agreement
  5. Missingness indicators (blank address, missing fields)
  6. Country agreement (open-set string equality)
  7. Script awareness (Latin vs Devanagari matching/mismatch)
"""

from typing import Dict, List, Set, Sequence, Any, Optional, Tuple
import math

FEATURE_NAMES: List[str] = [
    # 1. Name Features
    "name_core_exact",
    "name_norm_exact",
    "name_token_jaccard",
    "name_token_overlap_s1",
    "name_token_overlap_cand",
    "name_char_edit_similarity",
    "name_char_3gram_jaccard",
    "name_first_token_match",
    "name_len_diff_ratio",
    # 2. Address Features
    "address_norm_exact",
    "address_token_jaccard",
    "address_token_overlap_s1",
    "address_token_overlap_cand",
    "address_char_edit_similarity",
    "postal_match",
    "postal_both_present",
    # 3. Numeric Evidence (Mutually Exclusive)
    "numeric_perfect",
    "numeric_partial",
    "numeric_conflict",
    "numeric_none",
    # 4. Missingness & Length Differences
    "name_missing_s1",
    "name_missing_cand",
    "address_missing_s1",
    "address_missing_cand",
    "name_token_count_diff",
    "address_token_count_diff",
    # 5. Country, Source & Script
    "country_equal",
    "is_source_s2",
    "script_equal",
    "script_mismatch",
]


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Fast DP Levenshtein distance."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            curr[j + 1] = prev[j] if c1 == c2 else 1 + min(prev[j + 1], curr[j], prev[j])
        prev = curr
    return prev[len(s2)]


def _edit_similarity(s1: Optional[str], s2: Optional[str], max_len_cap: int = 48) -> float:
    """Normalized edit similarity in [0.0, 1.0]."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    s1_str = str(s1)[:max_len_cap]
    s2_str = str(s2)[:max_len_cap]
    if s1_str == s2_str:
        return 1.0
    l1 = len(s1_str)
    l2 = len(s2_str)
    max_len = max(l1, l2)
    if max_len == 0:
        return 1.0
    if abs(l1 - l2) / max_len >= 0.7:
        return 0.0
    dist = _levenshtein_distance(s1_str, s2_str)
    return max(0.0, 1.0 - (dist / max_len))


def _char_ngrams(text: Optional[str], n: int = 3) -> Set[str]:
    """Extract character n-grams from string."""
    if not text or len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(set1: Set[Any], set2: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return (intersection / union) if union > 0 else 0.0


def build_pair_features(s1: Any, cand: Any) -> Dict[str, float]:
    """Compute the fixed 30-dimensional feature dictionary for an (S1, Candidate) pair.

    `s1` and `cand` are NormalizedRecord instances from `normalize.py`.
    Returns:
        Dict mapping feature name -> float value.
    """
    feats: Dict[str, float] = {}

    # ---------------------------------------------------------
    # 1. Missingness & Field Availability
    # ---------------------------------------------------------
    s1_miss = getattr(s1, "missingness_flags", {}) or {}
    c_miss = getattr(cand, "missingness_flags", {}) or {}

    s1_name_missing = bool(s1_miss.get("name_missing", not bool(getattr(s1, "name_norm", ""))))
    c_name_missing = bool(c_miss.get("name_missing", not bool(getattr(cand, "name_norm", ""))))
    s1_addr_missing = bool(s1_miss.get("address_missing", not bool(getattr(s1, "address_norm", ""))))
    c_addr_missing = bool(c_miss.get("address_missing", not bool(getattr(cand, "address_norm", ""))))

    feats["name_missing_s1"] = 1.0 if s1_name_missing else 0.0
    feats["name_missing_cand"] = 1.0 if c_name_missing else 0.0
    feats["address_missing_s1"] = 1.0 if s1_addr_missing else 0.0
    feats["address_missing_cand"] = 1.0 if c_addr_missing else 0.0

    # ---------------------------------------------------------
    # 2. Name Features
    # ---------------------------------------------------------
    s1_norm_name = str(getattr(s1, "name_norm", "") or "")
    c_norm_name = str(getattr(cand, "name_norm", "") or "")
    s1_core_name = str(getattr(s1, "name_core", "") or "")
    c_core_name = str(getattr(cand, "name_core", "") or "")

    s1_name_toks = getattr(s1, "name_tokens", []) or []
    c_name_toks = getattr(cand, "name_tokens", []) or []

    feats["name_core_exact"] = 1.0 if (s1_core_name and c_core_name and s1_core_name == c_core_name) else 0.0
    feats["name_norm_exact"] = 1.0 if (s1_norm_name and c_norm_name and s1_norm_name == c_norm_name) else 0.0

    set_s1_toks = set(s1_name_toks)
    set_c_toks = set(c_name_toks)

    feats["name_token_jaccard"] = _jaccard(set_s1_toks, set_c_toks)
    inter_name = len(set_s1_toks & set_c_toks)
    feats["name_token_overlap_s1"] = (inter_name / len(set_s1_toks)) if set_s1_toks else 0.0
    feats["name_token_overlap_cand"] = (inter_name / len(set_c_toks)) if set_c_toks else 0.0

    # Character edit similarity on name (use fold if available for accent-robustness)
    s1_fold_name = getattr(s1, "name_fold", None) or s1_norm_name
    c_fold_name = getattr(cand, "name_fold", None) or c_norm_name
    feats["name_char_edit_similarity"] = _edit_similarity(s1_fold_name, c_fold_name)

    # 3-gram character Jaccard
    s1_grams = _char_ngrams(s1_fold_name, 3)
    c_grams = _char_ngrams(c_fold_name, 3)
    feats["name_char_3gram_jaccard"] = _jaccard(s1_grams, c_grams)

    # First token match
    feats["name_first_token_match"] = (
        1.0 if (s1_name_toks and c_name_toks and s1_name_toks[0] == c_name_toks[0]) else 0.0
    )

    # Name length difference ratio
    l1 = len(s1_norm_name)
    l2 = len(c_norm_name)
    max_l = max(l1, l2, 1)
    feats["name_len_diff_ratio"] = abs(l1 - l2) / max_l

    feats["name_token_count_diff"] = float(abs(len(s1_name_toks) - len(c_name_toks)))

    # ---------------------------------------------------------
    # 3. Address Features
    # ---------------------------------------------------------
    s1_norm_addr = str(getattr(s1, "address_norm", "") or "")
    c_norm_addr = str(getattr(cand, "address_norm", "") or "")

    s1_addr_toks = getattr(s1, "address_tokens", []) or []
    c_addr_toks = getattr(cand, "address_tokens", []) or []
    set_s1_addr = set(s1_addr_toks)
    set_c_addr = set(c_addr_toks)

    if s1_addr_missing or c_addr_missing or not s1_norm_addr or not c_norm_addr:
        feats["address_norm_exact"] = 0.0
        feats["address_token_jaccard"] = 0.0
        feats["address_token_overlap_s1"] = 0.0
        feats["address_token_overlap_cand"] = 0.0
        feats["address_char_edit_similarity"] = 0.0
        feats["address_token_count_diff"] = float(abs(len(s1_addr_toks) - len(c_addr_toks)))
    else:
        feats["address_norm_exact"] = 1.0 if (s1_norm_addr == c_norm_addr) else 0.0
        feats["address_token_jaccard"] = _jaccard(set_s1_addr, set_c_addr)
        inter_addr = len(set_s1_addr & set_c_addr)
        feats["address_token_overlap_s1"] = (inter_addr / len(set_s1_addr)) if set_s1_addr else 0.0
        feats["address_token_overlap_cand"] = (inter_addr / len(set_c_addr)) if set_c_addr else 0.0

        s1_fold_addr = getattr(s1, "address_fold", None) or s1_norm_addr
        c_fold_addr = getattr(cand, "address_fold", None) or c_norm_addr
        feats["address_char_edit_similarity"] = _edit_similarity(s1_fold_addr, c_fold_addr)
        feats["address_token_count_diff"] = float(abs(len(s1_addr_toks) - len(c_addr_toks)))

    # ---------------------------------------------------------
    # 4. Postal Code Features
    # ---------------------------------------------------------
    s1_postal = set(getattr(s1, "postal_tokens", []) or [])
    c_postal = set(getattr(cand, "postal_tokens", []) or [])

    if s1_postal and c_postal:
        feats["postal_both_present"] = 1.0
        feats["postal_match"] = 1.0 if bool(s1_postal & c_postal) else 0.0
    else:
        feats["postal_both_present"] = 0.0
        feats["postal_match"] = 0.0

    # ---------------------------------------------------------
    # 5. Numeric Evidence (Mutually Exclusive Categories)
    # ---------------------------------------------------------
    s1_nums = set(getattr(s1, "numeric_tokens", []) or [])
    c_nums = set(getattr(cand, "numeric_tokens", []) or [])

    # Default all 4 indicators to 0.0
    feats["numeric_perfect"] = 0.0
    feats["numeric_partial"] = 0.0
    feats["numeric_conflict"] = 0.0
    feats["numeric_none"] = 0.0

    if not s1_nums or not c_nums:
        feats["numeric_none"] = 1.0
    elif s1_nums == c_nums:
        feats["numeric_perfect"] = 1.0
    elif bool(s1_nums & c_nums):
        feats["numeric_partial"] = 1.0
    else:
        feats["numeric_conflict"] = 1.0

    # ---------------------------------------------------------
    # 6. Country & Source
    # ---------------------------------------------------------
    s1_country = str(getattr(s1, "country_norm", "") or "").lower().strip()
    c_country = str(getattr(cand, "country_norm", "") or "").lower().strip()
    feats["country_equal"] = 1.0 if (s1_country and c_country and s1_country == c_country) else 0.0

    cand_src = str(getattr(cand, "source", "") or "")
    if not cand_src and hasattr(cand, "entity_id"):
        cand_src = cand.entity_id[:2]
    feats["is_source_s2"] = 1.0 if cand_src.upper() == "S2" else 0.0

    # ---------------------------------------------------------
    # 7. Script Features
    # ---------------------------------------------------------
    s1_scripts = getattr(s1, "script_flags", {}) or {}
    c_scripts = getattr(cand, "script_flags", {}) or {}

    s1_name_script = (s1_scripts.get("name") or {}).get("primary_script", "latin")
    c_name_script = (c_scripts.get("name") or {}).get("primary_script", "latin")

    if s1_name_script and c_name_script:
        feats["script_equal"] = 1.0 if (s1_name_script == c_name_script) else 0.0
        is_cross = (
            (s1_name_script == "latin" and c_name_script == "devanagari")
            or (s1_name_script == "devanagari" and c_name_script == "latin")
        )
        feats["script_mismatch"] = 1.0 if is_cross else 0.0
    else:
        feats["script_equal"] = 1.0
        feats["script_mismatch"] = 0.0

    return feats


def build_feature_vector(s1: Any, cand: Any) -> List[float]:
    """Return ordered feature vector matching FEATURE_NAMES exactly."""
    feat_dict = build_pair_features(s1, cand)
    return [feat_dict[k] for k in FEATURE_NAMES]


def build_pair_features_batch(
    candidate_map: Dict[str, Sequence[str]],
    s1_lookup: Dict[str, Any],
    cand_lookup: Dict[str, Any],
) -> Tuple[List[List[float]], List[Tuple[str, str]]]:
    """Generate feature vectors for all candidate pairs in candidate_map.

    Returns:
        (X, pair_ids)
        where X is a list of feature vectors and pair_ids is a list of (s1_id, cand_id) tuples.
    """
    X: List[List[float]] = []
    pair_ids: List[Tuple[str, str]] = []

    for s1_id, cands in candidate_map.items():
        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            continue
        for cid in cands:
            cand_rec = cand_lookup.get(cid)
            if cand_rec is None:
                continue
            feats = build_feature_vector(s1_rec, cand_rec)
            X.append(feats)
            pair_ids.append((s1_id, cid))

    return X, pair_ids
