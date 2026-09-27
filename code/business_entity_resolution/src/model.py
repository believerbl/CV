"""
model.py
========
Machine learning candidate-pair matcher module for Amazon ML Challenge 2026.
Owned by Parimarjan Shukla.

Consumes 30-dimensional engineered pair features from `features.py`
and predicts candidate-level match confidence scores for `decision.py`.

Core Responsibilities:
  1. build_training_dataset(...)
     Constructs labeled training pairs (X, y, pair_ids) from candidate_map
     and ground truth mapping, with optional deterministic negative balancing.
  2. train_model(...)
     Fits a fast, deterministic gradient-boosted tree (HistGradientBoostingClassifier)
     or random forest on feature vectors.
  3. predict_scores(...) / predict_pair_scores(...)
     Produces candidate-level confidence scores in [0.0, 1.0] for every
     candidate pair (both structured dict s1_id -> {cand_id: score} and flat array).
  4. save_model(...) / load_model(...)
     Deterministically saves/loads model weights and feature schema metadata
     to ensure bit-for-bit reproducible predictions.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import joblib
except ImportError:
    joblib = None
import pickle

from features import FEATURE_NAMES, build_feature_vector

EXPECTED_FEATURE_DIM = len(FEATURE_NAMES)  # 30


# =============================================================================
# 1. Training Data Construction
# =============================================================================

def build_training_dataset(
    candidate_map: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Set[str]],
    s1_lookup: Dict[str, Any],
    cand_lookup: Dict[str, Any],
    balance_ratio: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[List[List[float]], List[int], List[Tuple[str, str]]]:
    """Construct labeled training pairs from blocker-generated candidate pairs.

    Label Rule (per SRS):
      - 1 (positive) if cand_id is in ground_truth.get(s1_id, set())
      - 0 (negative) otherwise

    Only candidate pairs present in `candidate_map` are used (the operational space).
    Negative pairs can be deterministically subsampled if `balance_ratio` is specified
    (e.g., balance_ratio=3.0 keeps at most 3 negatives per positive).

    Args:
        candidate_map: Dict mapping s1_id -> sequence of candidate IDs.
        ground_truth: Dict mapping s1_id -> set of true matching IDs.
        s1_lookup: Dict mapping s1_id -> NormalizedRecord.
        cand_lookup: Dict mapping cand_id -> NormalizedRecord.
        balance_ratio: Optional ratio of negatives to positives (e.g., 3.0 or 5.0).
        random_state: Seed for deterministic subsampling and pair ordering.

    Returns:
        (X, y, pair_ids)
        - X: List of 30-dim feature vectors
        - y: List of integer labels (0 or 1)
        - pair_ids: List of (s1_id, cand_id) tuples
    """
    pos_items: List[Tuple[List[float], int, Tuple[str, str]]] = []
    neg_items: List[Tuple[List[float], int, Tuple[str, str]]] = []

    # Sort keys for determinism across platforms
    sorted_s1_ids = sorted(candidate_map.keys())

    for s1_id in sorted_s1_ids:
        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            continue

        true_matches = ground_truth.get(s1_id, set())
        candidates = candidate_map[s1_id]

        for cand_id in candidates:
            cand_rec = cand_lookup.get(cand_id)
            if cand_rec is None:
                continue

            feats = build_feature_vector(s1_rec, cand_rec)
            label = 1 if cand_id in true_matches else 0
            item = (feats, label, (s1_id, cand_id))

            if label == 1:
                pos_items.append(item)
            else:
                neg_items.append(item)

    # Balance negatives if requested
    if balance_ratio is not None and len(pos_items) > 0:
        max_neg = int(math.ceil(len(pos_items) * float(balance_ratio)))
        if len(neg_items) > max_neg:
            rng = random.Random(random_state)
            neg_items = rng.sample(neg_items, max_neg)

    combined = pos_items + neg_items
    rng_shuffle = random.Random(random_state)
    rng_shuffle.shuffle(combined)

    X: List[List[float]] = []
    y: List[int] = []
    pair_ids: List[Tuple[str, str]] = []

    for feats, label, pid in combined:
        X.append(feats)
        y.append(label)
        pair_ids.append(pid)

    return X, y, pair_ids


# =============================================================================
# 2. ER Model Wrapper
# =============================================================================

class ERModel:
    """Deterministic Entity Resolution Pair Matcher Model.

    Wraps a scikit-learn classifier with fixed 30-feature validation,
    deterministic scoring, and metadata preservation.
    """

    def __init__(
        self,
        estimator: Any,
        feature_names: Optional[List[str]] = None,
        model_type: str = "hist_gbdt",
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        version: str = "1.0.0",
    ):
        self.estimator = estimator
        self.feature_names = list(feature_names if feature_names is not None else FEATURE_NAMES)
        self.expected_dim = len(self.feature_names)
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.random_state = random_state
        self.version = version

    def _validate_features(self, X: Any) -> Any:
        """Validate input feature matrix shape and type."""
        if X is None:
            raise ValueError("Input feature matrix X cannot be None.")

        if np is not None and isinstance(X, np.ndarray):
            X_arr = X
        else:
            try:
                if np is not None:
                    X_arr = np.asarray(X, dtype=float)
                else:
                    X_arr = [list(map(float, row)) for row in X]
            except (ValueError, TypeError) as e:
                raise ValueError(f"Malformed feature matrix: could not convert elements to float: {e}") from e

        if np is not None and isinstance(X_arr, np.ndarray):
            if X_arr.ndim == 0:
                raise ValueError("Expected 2D array of shape (N, 30), got scalar.")
            if X_arr.ndim == 1:
                if X_arr.shape[0] == 0:
                    return X_arr.reshape(0, self.expected_dim)
                raise ValueError(
                    f"Expected 2D array of shape (N, {self.expected_dim}), got 1D array of shape {X_arr.shape}."
                )
            if X_arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got array with {X_arr.ndim} dimensions.")
            if X_arr.shape[1] != self.expected_dim:
                raise ValueError(
                    f"Feature dimension mismatch: model expects {self.expected_dim} features, got {X_arr.shape[1]}."
                )
            return X_arr
        else:
            # Fallback when numpy is unavailable
            if len(X_arr) == 0:
                return X_arr
            for i, row in enumerate(X_arr):
                if len(row) != self.expected_dim:
                    raise ValueError(
                        f"Feature dimension mismatch in row {i}: expected {self.expected_dim} features, got {len(row)}."
                    )
            return X_arr

    def predict_proba(self, X: Any) -> Any:
        """Predict positive-class match probabilities for feature rows.

        Returns 1D array of probabilities in [0.0, 1.0].
        """
        X_valid = self._validate_features(X)

        if (np is not None and isinstance(X_valid, np.ndarray) and len(X_valid) == 0) or (
            isinstance(X_valid, list) and len(X_valid) == 0
        ):
            return np.array([], dtype=float) if np is not None else []

        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X_valid)
            # If binary classifier with classes [0, 1]
            if hasattr(self.estimator, "classes_") and len(self.estimator.classes_) == 2:
                pos_idx = list(self.estimator.classes_).index(1) if 1 in self.estimator.classes_ else 1
                return probs[:, pos_idx]
            elif hasattr(probs, "shape") and len(probs.shape) == 2 and probs.shape[1] >= 2:
                return probs[:, 1]
            elif hasattr(probs, "shape") and len(probs.shape) == 2 and probs.shape[1] == 1:
                return probs[:, 0]
            else:
                return np.asarray(probs, dtype=float).ravel()
        elif hasattr(self.estimator, "decision_function"):
            raw_scores = self.estimator.decision_function(X_valid)
            # Apply sigmoid
            if np is not None:
                return 1.0 / (1.0 + np.exp(-raw_scores))
            return [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]
        elif hasattr(self.estimator, "predict"):
            preds = self.estimator.predict(X_valid)
            return np.asarray(preds, dtype=float) if np is not None else [float(p) for p in preds]
        else:
            raise TypeError(f"Underlying estimator {type(self.estimator)} has no predict_proba or predict method.")

    def predict_pair_scores(self, X: Any) -> Any:
        """Alias for predict_proba."""
        return self.predict_proba(X)

    def predict_scores(
        self,
        candidate_map: Dict[str, Sequence[str]],
        s1_lookup: Dict[str, Any],
        cand_lookup: Dict[str, Any],
        batch_size: int = 10000,
    ) -> Dict[str, Dict[str, float]]:
        """Predict confidence scores for all pairs in candidate_map.

        Returns:
            Dict mapping s1_id -> {cand_id: score_float}
            Every candidate in candidate_map receives a score in [0.0, 1.0].
            If an s1_id has no candidates, scores[s1_id] = {}.
        """
        result: Dict[str, Dict[str, float]] = {s1_id: {} for s1_id in candidate_map}

        if not candidate_map:
            return result

        batch_X: List[List[float]] = []
        batch_pairs: List[Tuple[str, str]] = []

        def flush_batch():
            nonlocal batch_X, batch_pairs
            if not batch_X:
                return
            scores = self.predict_proba(batch_X)
            for (sid, cid), score in zip(batch_pairs, scores):
                result[sid][cid] = float(score)
            batch_X = []
            batch_pairs = []

        for s1_id, candidates in candidate_map.items():
            s1_rec = s1_lookup.get(s1_id)
            for cand_id in candidates:
                cand_rec = cand_lookup.get(cand_id)
                if s1_rec is None or cand_rec is None:
                    # If either record is unresolvable, assign 0.0 confidence
                    result[s1_id][cand_id] = 0.0
                    continue

                feats = build_feature_vector(s1_rec, cand_rec)
                batch_X.append(feats)
                batch_pairs.append((s1_id, cand_id))

                if len(batch_X) >= batch_size:
                    flush_batch()

        flush_batch()
        return result

    def save(self, filepath: str) -> None:
        """Save this model deterministically to filepath."""
        save_model(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> ERModel:
        """Load an ERModel from filepath."""
        return load_model(filepath)


# =============================================================================
# 3. Model Training
# =============================================================================

def train_model(
    X: Any,
    y: Any,
    model_type: str = "hist_gbdt",
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> ERModel:
    """Train a fast gradient-boosted tree or tree-based matcher model.

    Args:
        X: Sequence of 30-dim float feature vectors.
        y: Sequence of binary labels (0 or 1).
        model_type: "hist_gbdt" (fast histogram-based GBDT) or "random_forest".
        hyperparameters: Optional dictionary of hyperparameter overrides.
        random_state: Integer seed for deterministic training.

    Returns:
        Fitted ERModel instance.
    """
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for train_model. Please install scikit-learn in your environment."
        )

    if X is None or y is None:
        raise ValueError("X and y cannot be None.")

    try:
        X_arr = np.asarray(X, dtype=float) if np is not None else list(X)
        y_arr = np.asarray(y, dtype=int) if np is not None else list(y)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Malformed training data: could not convert to numeric arrays: {e}") from e

    n_samples = len(X_arr)
    if n_samples == 0:
        raise ValueError("Cannot train model on empty training data (0 samples).")

    if len(y_arr) != n_samples:
        raise ValueError(f"Mismatched sample counts: X has {n_samples} rows, y has {len(y_arr)} labels.")

    if np is not None and isinstance(X_arr, np.ndarray):
        if X_arr.ndim != 2 or X_arr.shape[1] != EXPECTED_FEATURE_DIM:
            dim = X_arr.shape[1] if X_arr.ndim == 2 else f"{X_arr.ndim}D"
            raise ValueError(
                f"Feature dimension mismatch: expected {EXPECTED_FEATURE_DIM} features, got {dim}."
            )
    else:
        for i, row in enumerate(X_arr):
            if len(row) != EXPECTED_FEATURE_DIM:
                raise ValueError(
                    f"Feature dimension mismatch in row {i}: expected {EXPECTED_FEATURE_DIM} features, got {len(row)}."
                )

    params = dict(hyperparameters or {})
    unique_labels = np.unique(y_arr) if np is not None else list(set(y_arr))

    # Single-class fallback: if only 1 class is present, use DummyClassifier
    if len(unique_labels) < 2:
        estimator = DummyClassifier(strategy="constant", constant=unique_labels[0])
        estimator.fit(X_arr, y_arr)
        return ERModel(
            estimator=estimator,
            feature_names=FEATURE_NAMES,
            model_type="dummy_single_class",
            hyperparameters={"constant": int(unique_labels[0])},
            random_state=random_state,
        )

    if model_type == "hist_gbdt":
        hist_params = {
            "max_iter": params.get("max_iter", 100),
            "max_leaf_nodes": params.get("max_leaf_nodes", 31),
            "min_samples_leaf": params.get("min_samples_leaf", 20),
            "learning_rate": params.get("learning_rate", 0.1),
            "l2_regularization": params.get("l2_regularization", 0.0),
            "random_state": random_state,
        }
        if "class_weight" in params:
            hist_params["class_weight"] = params["class_weight"]

        estimator = HistGradientBoostingClassifier(**hist_params)

    elif model_type == "random_forest":
        rf_params = {
            "n_estimators": params.get("n_estimators", 100),
            "max_depth": params.get("max_depth", 15),
            "min_samples_leaf": params.get("min_samples_leaf", 5),
            "random_state": random_state,
            "n_jobs": params.get("n_jobs", -1),
        }
        if "class_weight" in params:
            rf_params["class_weight"] = params["class_weight"]

        estimator = RandomForestClassifier(**rf_params)

    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose 'hist_gbdt' or 'random_forest'."
        )

    estimator.fit(X_arr, y_arr)

    return ERModel(
        estimator=estimator,
        feature_names=FEATURE_NAMES,
        model_type=model_type,
        hyperparameters=params,
        random_state=random_state,
    )


# =============================================================================
# 4. Predict Scores Convenience Function
# =============================================================================

def predict_scores(
    model: Union[ERModel, Any],
    candidate_map: Dict[str, Sequence[str]],
    s1_lookup: Dict[str, Any],
    cand_lookup: Dict[str, Any],
    batch_size: int = 10000,
) -> Dict[str, Dict[str, float]]:
    """Predict candidate-level confidence scores for all candidate pairs.

    Args:
        model: ERModel instance or fitted estimator.
        candidate_map: Dict mapping s1_id -> sequence of candidate IDs.
        s1_lookup: Dict mapping s1_id -> NormalizedRecord.
        cand_lookup: Dict mapping cand_id -> NormalizedRecord.
        batch_size: Chunk size for feature batching.

    Returns:
        Dict mapping s1_id -> {cand_id: score}
    """
    if isinstance(model, ERModel):
        return model.predict_scores(candidate_map, s1_lookup, cand_lookup, batch_size=batch_size)
    else:
        # Wrap raw estimator
        wrapper = ERModel(estimator=model)
        return wrapper.predict_scores(candidate_map, s1_lookup, cand_lookup, batch_size=batch_size)


def predict_pair_scores(model: Union[ERModel, Any], X: Any) -> Any:
    """Predict scores for a raw 2D feature matrix X.

    Args:
        model: ERModel instance or fitted estimator.
        X: Sequence of 30-dim feature vectors.

    Returns:
        1D array of match probabilities.
    """
    if isinstance(model, ERModel):
        return model.predict_proba(X)
    else:
        wrapper = ERModel(estimator=model)
        return wrapper.predict_proba(X)


# =============================================================================
# 5. Deterministic Serialization & Deserialization
# =============================================================================

def save_model(model: ERModel, filepath: str) -> None:
    """Save an ERModel deterministically to disk with schema metadata.

    Args:
        model: ERModel instance to serialize.
        filepath: Target file path (e.g., 'model.joblib' or 'model.pkl').
    """
    if not isinstance(model, ERModel):
        raise TypeError(f"Expected ERModel instance, got {type(model)}.")

    payload = {
        "format": "AmazonML_ERModel",
        "version": model.version,
        "model_type": model.model_type,
        "feature_names": model.feature_names,
        "expected_dim": model.expected_dim,
        "hyperparameters": model.hyperparameters,
        "random_state": model.random_state,
        "estimator": model.estimator,
    }

    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if joblib is not None:
        joblib.dump(payload, filepath, compress=3)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(filepath: str) -> ERModel:
    """Load an ERModel from disk and verify schema integrity.

    Args:
        filepath: Path to saved model file.

    Returns:
        Reconstructed ERModel instance.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    if joblib is not None:
        try:
            payload = joblib.load(filepath)
        except Exception:
            with open(filepath, "rb") as f:
                payload = pickle.load(f)
    else:
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

    if not isinstance(payload, dict) or "estimator" not in payload:
        raise ValueError(f"Corrupted or invalid model file: {filepath}")

    saved_features = payload.get("feature_names", [])
    if saved_features != FEATURE_NAMES:
        raise ValueError(
            f"Feature schema mismatch in loaded model! Expected {len(FEATURE_NAMES)} features, "
            f"found {len(saved_features)}."
        )

    return ERModel(
        estimator=payload["estimator"],
        feature_names=saved_features,
        model_type=payload.get("model_type", "unknown"),
        hyperparameters=payload.get("hyperparameters", {}),
        random_state=payload.get("random_state", 42),
        version=payload.get("version", "1.0.0"),
    )
