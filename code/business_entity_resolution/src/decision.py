"""
decision.py
===========
Set-decision layer for Amazon ML Challenge 2026 (Business Entity Resolution).
Owned by Parimarjan Shukla.

Transforms continuous candidate match confidence scores produced by `model.py`
into variable-sized match sets (zero, one, or many) per S1 entity.

Contract:
  candidate scores
        ↓
  sort deterministically
        ↓
  threshold tau (keep candidates with score >= tau)
        ↓
  numeric contradiction handling (optional suppression)
        ↓
  empty-set gate (allow [] when no candidate qualifies)
        ↓
  optional set-size cap (max_matches_per_s1)
        ↓
  final match IDs (zero, one, or many)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    from evaluate import evaluate_predictions, EvaluationReport
    HAS_EVALUATE = True
except ImportError:
    HAS_EVALUATE = False


@dataclass
class DecisionConfig:
    """Configuration parameters for the decision layer."""
    threshold: float = 0.5
    singleton_threshold: Optional[float] = None
    suppress_numeric_conflict: bool = False
    max_matches_per_s1: Optional[int] = None
    output_order: str = "id_asc"  # "id_asc" or "score_desc"


def _validate_score(cand_id: str, score: Any) -> float:
    """Validate that score is a valid finite float in [0.0, 1.0]."""
    try:
        val = float(score)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Malformed candidate score for '{cand_id}': expected float, got {type(score).__name__} ({score!r})"
        ) from e

    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"Invalid score value for '{cand_id}': {val}")
    if val < 0.0 or val > 1.0:
        raise ValueError(f"Score for '{cand_id}' out of bounds [0.0, 1.0]: {val}")
    return val


def select_matches_for_entity(
    candidate_scores: Dict[str, float],
    threshold: float = 0.5,
    singleton_threshold: Optional[float] = None,
    numeric_conflicts: Optional[Set[str]] = None,
    suppress_numeric_conflict: bool = False,
    max_matches: Optional[int] = None,
    output_order: str = "id_asc",
) -> List[str]:
    """Select final match entity IDs for a single S1 entity from its candidate scores.

    Args:
        candidate_scores: Dict mapping cand_id -> confidence score in [0.0, 1.0].
        threshold: Minimum score required to accept a candidate pair.
        singleton_threshold: Optional gate; if max(score) < singleton_threshold,
            the entity is classified as an empty match set ([]).
        numeric_conflicts: Optional set of candidate IDs that have address numeric conflict.
        suppress_numeric_conflict: If True, candidates in numeric_conflicts are filtered out.
        max_matches: Optional cap on the maximum number of matches retained.
        output_order: "id_asc" (default, alphabetical) or "score_desc" (ranked).

    Returns:
        Deterministic list of selected candidate IDs (can be empty: []).
    """
    if candidate_scores is None:
        raise ValueError("candidate_scores cannot be None.")

    try:
        t_val = float(threshold)
    except (ValueError, TypeError) as e:
        raise ValueError(f"threshold must be numeric, got {threshold!r}") from e

    if math.isnan(t_val) or t_val < 0.0 or t_val > 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {t_val}")

    if not candidate_scores:
        return []

    # Validate all scores
    validated: List[Tuple[str, float]] = []
    for cid, s in candidate_scores.items():
        score_val = _validate_score(cid, s)
        validated.append((cid, score_val))

    # Empty-set / Singleton gate:
    # If the maximum score among all candidates does not reach the gate, predict []
    if singleton_threshold is not None:
        st_val = float(singleton_threshold)
        max_score = max(s for _, s in validated)
        if max_score < st_val:
            return []

    # Filter candidates meeting global threshold
    qualifying: List[Tuple[str, float]] = []
    for cid, s in validated:
        if s >= t_val:
            # Numeric conflict suppression
            if suppress_numeric_conflict and numeric_conflicts and cid in numeric_conflicts:
                continue
            qualifying.append((cid, s))

    if not qualifying:
        return []

    # Sort deterministically: highest score first, then tie-break on candidate ID
    qualifying.sort(key=lambda item: (-item[1], item[0]))

    # Apply optional maximum set-size cap
    if max_matches is not None and max_matches > 0:
        qualifying = qualifying[:max_matches]

    # Deduplicate candidate IDs while preserving order
    seen: Set[str] = set()
    retained_ids: List[str] = []
    for cid, _ in qualifying:
        if cid not in seen:
            seen.add(cid)
            retained_ids.append(cid)

    # Format final output order
    if output_order == "id_asc":
        retained_ids.sort()
    elif output_order == "score_desc":
        pass  # already sorted by (-score, cid)
    else:
        raise ValueError(f"Unsupported output_order: '{output_order}'. Choose 'id_asc' or 'score_desc'.")

    return retained_ids


def make_decisions(
    scores: Dict[str, Dict[str, float]],
    config: Optional[DecisionConfig] = None,
    threshold: float = 0.5,
    singleton_threshold: Optional[float] = None,
    numeric_conflict_map: Optional[Dict[str, Set[str]]] = None,
    suppress_numeric_conflict: bool = False,
    max_matches_per_s1: Optional[int] = None,
    output_order: str = "id_asc",
) -> Dict[str, List[str]]:
    """Batch decision interface for end-to-end pipeline.

    Maps every S1 entity to its selected match IDs.

    Args:
        scores: Dict mapping s1_id -> {cand_id: score}
        config: Optional DecisionConfig dataclass (overrides individual arguments if provided)
        threshold: Score threshold tau
        singleton_threshold: Optional empty-set gate threshold
        numeric_conflict_map: Dict mapping s1_id -> set of candidate IDs with numeric conflicts
        suppress_numeric_conflict: Whether to suppress candidates with numeric conflicts
        max_matches_per_s1: Optional cap on matches per S1
        output_order: Ordering of matched IDs ("id_asc" or "score_desc")

    Returns:
        Dict mapping s1_id -> List of matched IDs.
    """
    if scores is None:
        raise ValueError("scores dictionary cannot be None.")

    # Unpack config if supplied
    if config is not None:
        threshold = config.threshold
        singleton_threshold = config.singleton_threshold
        suppress_numeric_conflict = config.suppress_numeric_conflict
        max_matches_per_s1 = config.max_matches_per_s1
        output_order = config.output_order

    decisions: Dict[str, List[str]] = {}

    for s1_id, cand_scores in scores.items():
        conflicts = numeric_conflict_map.get(s1_id) if numeric_conflict_map else None
        matches = select_matches_for_entity(
            candidate_scores=cand_scores,
            threshold=threshold,
            singleton_threshold=singleton_threshold,
            numeric_conflicts=conflicts,
            suppress_numeric_conflict=suppress_numeric_conflict,
            max_matches=max_matches_per_s1,
            output_order=output_order,
        )
        decisions[s1_id] = matches

    return decisions


def sweep_thresholds(
    scores: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, Set[str]],
    thresholds: Sequence[float] = (
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ),
    singleton_threshold: Optional[float] = None,
    numeric_conflict_map: Optional[Dict[str, Set[str]]] = None,
    suppress_numeric_conflict: bool = False,
    max_matches_per_s1: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Perform a systematic threshold sweep on candidate scores against ground truth.

    Computes for each threshold tau:
      - Macro F0.5
      - Macro Precision
      - Macro Recall
      - Candidate Recall
      - Empty-set rate (% of S1 predicted empty)
      - Singleton performance (accuracy on true singletons)
      - False-positive count
      - Average predicted set size

    Returns:
        List of dicts containing the measured metrics per threshold.
    """
    if not HAS_EVALUATE:
        raise ImportError("evaluate.py is required for sweep_thresholds.")

    # Calculate total true positive links for candidate recall baseline
    total_true_links = sum(len(matches) for matches in ground_truth.values())
    total_candidates_pool: Set[Tuple[str, str]] = set()
    for sid, cands in scores.items():
        for cid in cands:
            total_candidates_pool.add((sid, cid))

    cand_tp = 0
    for sid, true_matches in ground_truth.items():
        for mid in true_matches:
            if (sid, mid) in total_candidates_pool:
                cand_tp += 1
    candidate_recall = (cand_tp / total_true_links) if total_true_links > 0 else 1.0

    results: List[Dict[str, Any]] = []

    for tau in thresholds:
        pred_dict = make_decisions(
            scores=scores,
            threshold=tau,
            singleton_threshold=singleton_threshold,
            numeric_conflict_map=numeric_conflict_map,
            suppress_numeric_conflict=suppress_numeric_conflict,
            max_matches_per_s1=max_matches_per_s1,
            output_order="id_asc",
        )

        pred_sets: Dict[str, Set[str]] = {sid: set(matches) for sid, matches in pred_dict.items()}

        # Evaluate official entity-level Macro F0.5
        report: EvaluationReport = evaluate_predictions(ground_truth, pred_sets)

        n_s1 = len(ground_truth)
        n_empty_pred = sum(1 for sid in ground_truth if len(pred_sets.get(sid, set())) == 0)
        empty_set_rate = (n_empty_pred / n_s1) if n_s1 > 0 else 0.0

        total_pred_matches = sum(len(m) for m in pred_dict.values())
        avg_set_size = (total_pred_matches / n_s1) if n_s1 > 0 else 0.0

        fp_count = report.total_pred_links - report.total_tp_links

        results.append({
            "threshold": round(float(tau), 4),
            "macro_f05": report.macro_f05,
            "macro_precision": report.macro_precision,
            "macro_recall": report.macro_recall,
            "candidate_recall": round(candidate_recall, 4),
            "empty_set_rate": round(empty_set_rate, 4),
            "singleton_accuracy": report.singleton_accuracy,
            "false_positive_count": fp_count,
            "average_predicted_set_size": round(avg_set_size, 4),
            "total_pred_links": report.total_pred_links,
            "total_tp_links": report.total_tp_links,
        })

    return results
