"""
blocking.py
============
Candidate Generation / Blocking module (Piyush) — Amazon ML Challenge 2026,
Business Entity Resolution.

This module NEVER re-normalizes anything. It consumes NormalizedRecord
objects produced by Nidhi's `normalize.py` (the single shared normalization
contract) via `normalize_dataframe_rows()`. Do not import or duplicate any
of normalize.py's internal string-processing logic here.

Currently implemented: Pass A — Exact / Core Name Blocking,
Pass B — Rare/IDF Token Blocking, Pass C — Numeric/Address Anchor
Blocking, Pass D — Character 3/4-gram Blocking (typo/noise
tolerant), and Pass E — Script-Aware Blocking.

Public API
----------
generate_candidates(s1_records, s2_records, s3_records, config) -> Dict[str, List[str]]

    s1_records, s2_records, s3_records:
        Lists of raw row dicts with keys entity_id, business_name,
        business_address, country (as loaded from TSV), OR lists of
        already-built NormalizedRecord objects. Either is accepted;
        raw dicts are passed through normalize_dataframe_rows().

    config:
        Dict of blocking parameters. Recognized keys for Pass A:
            "use_name_norm_exact": bool (default True)
                Whether to also build an exact name_norm index
                (in addition to the name_core index).
        Recognized keys for Pass D (character n-gram blocking):
            "pass_d_enabled": bool (default True)
            "pass_d_ngram_sizes": tuple/list of int (default (3, 4))
            "pass_d_max_document_frequency": int (default 100)
                Absolute cap on a gram's S2+S3 document frequency.
            "pass_d_max_document_fraction": float (default 0.05)
                Relative cap; effective threshold is
                min(max_abs, max(2, int(fraction * N))).
            "pass_d_max_grams_per_record": int (default 30)
                Query-side cap: use only the rarest K grams per S1.
            "pass_d_min_shared_grams": int (default 2)
                Minimum distinctive shared grams to emit a candidate.
        Recognized keys for Pass B (rare/IDF token blocking):
            "pass_b_enabled": bool (default True)
            "pass_b_min_token_len": int (default 2)
            "pass_b_max_df_floor": int (default 50)
            "pass_b_max_df_ratio": float (default 0.01)
            "pass_b_top_k_tokens": int (default 3)
            "pass_b_max_candidates": int (default 200)
        Recognized keys for Pass C (numeric/address anchor blocking):
            "pass_c_enabled": bool (default True)
            "pass_c_use_postal": bool (default True)
            "pass_c_use_numeric": bool (default True)
            "pass_c_use_address_tokens": bool (default False)
            "pass_c_min_numeric_len": int (default 3)
            "pass_c_min_address_token_len": int (default 4)
            "pass_c_max_df_floor": int (default 50)
            "pass_c_max_df_ratio": float (default 0.005)
            "pass_c_top_k_anchors": int (default 3)
            "pass_c_max_candidates": int (default 200)
        Recognized keys for Pass E (script-aware blocking):
            "pass_e_enabled": bool (default True)
            "pass_e_use_mixed_expansion": bool (default True)
            "pass_e_min_token_len": int (default 2)
            "pass_e_max_df_floor": int (default 50)
            "pass_e_max_df_ratio": float (default 0.01)
            "pass_e_top_k_tokens": int (default 3)
            "pass_e_max_candidates": int (default 200)
            "pass_e_max_bucket_size": int (default 500)

    Returns:
        candidate_map: Dict[str, List[str]]
            - Contains an entry for every S1 record (entity_id -> list),
              even if the list is empty.
            - Values contain only S2/S3 entity_ids (never S1 ids, never
              duplicates).
            - Candidate ordering within each list is deterministic.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

from normalize import normalize_dataframe_rows, NormalizedRecord, is_blank

# A "record" fed into this module is either a raw dict (from TSV loading)
# or an already-normalized record. We normalize dicts on the way in and
# pass NormalizedRecord objects straight through, so callers upstream
# (e.g. a pipeline that already normalized once) never pay for re-work.
RawOrNormalized = Union[dict, NormalizedRecord]


def _ensure_normalized(records: Sequence[RawOrNormalized]) -> List[NormalizedRecord]:
    """Return a List[NormalizedRecord], normalizing raw dicts via Nidhi's
    normalize_dataframe_rows() where needed. Never re-implements
    normalization; only decides whether normalize_dataframe_rows() needs
    to be called at all.
    """
    if not records:
        return []

    # Mixed lists are not expected in this pipeline, but handle the
    # common case cheaply: check the first element's type only, since
    # per-record type dispatch would be O(n) constant-factor overhead
    # across the entire multi-million-row corpus for no reason.
    if isinstance(records[0], NormalizedRecord):
        return list(records)  # already normalized upstream

    return normalize_dataframe_rows(records)


def _build_name_index(
    records: List[NormalizedRecord], key_field: str
) -> Dict[str, List[str]]:
    """Build an inverted index: normalized-name-key -> [entity_id, ...].

    key_field is either "name_core" or "name_norm". Blank/None keys are
    never indexed (a blank name must not become a single giant bucket
    that fans every blank-name S1 record out against every blank-name
    S2/S3 record).

    Complexity: O(N) over the given records, one dict insert per record.
    """
    index: Dict[str, List[str]] = {}
    for rec in records:
        key = getattr(rec, key_field)
        if is_blank(key):
            continue
        index.setdefault(key, []).append(rec.entity_id)
    return index


def _pass_a_exact_core_name(
    s1_norm: List[NormalizedRecord],
    s2_norm: List[NormalizedRecord],
    s3_norm: List[NormalizedRecord],
    config: dict,
) -> Dict[str, List[str]]:
    """Pass A: Exact / Core Name Blocking.

    Builds inverted indexes over S2 and S3 keyed by `name_core`
    (legal-form-stripped identity signal), and optionally also by
    `name_norm` (exact full-name match, legal form included) — this
    catches true exact-string duplicates that name_core blocking alone
    would still catch, but keeping both means a record whose name_core
    collapses two genuinely different brands sharing a legal-form-free
    remainder is not the only path to a match; name_norm gives an
    additional, stricter exact-match signal layered on top.

    For every S1 record, look up its name_core (and, if enabled,
    name_norm) in the S2/S3 indexes — O(1) dict lookups, not a scan of
    S2/S3 — and union the resulting candidate ids, de-duplicated and
    deterministically ordered.

    Blank-name S1 records get an empty candidate list from this pass
    (never crash, never become a giant bucket) — later passes (B/C/D/E)
    are expected to be the ones that can still find candidates for them
    via address/token signals; Pass A cannot help a nameless record by
    definition.
    """
    use_name_norm_exact = config.get("use_name_norm_exact", True)

    # Build the two indexes once, over S2+S3 combined so a single lookup
    # per S1 record suffices instead of two lookups (one per source).
    s2_s3_norm = s2_norm + s3_norm  # cheap: list of references, not copies

    core_index = _build_name_index(s2_s3_norm, "name_core")
    norm_index: Dict[str, List[str]] = {}
    if use_name_norm_exact:
        norm_index = _build_name_index(s2_s3_norm, "name_norm")

    candidate_map: Dict[str, List[str]] = {}

    for rec in s1_norm:
        candidates: List[str] = []
        seen: set = set()

        core_key = rec.name_core
        if not is_blank(core_key):
            for cid in core_index.get(core_key, []):
                if cid not in seen:
                    seen.add(cid)
                    candidates.append(cid)

        if use_name_norm_exact:
            norm_key = rec.name_norm
            if not is_blank(norm_key):
                for cid in norm_index.get(norm_key, []):
                    if cid not in seen:
                        seen.add(cid)
                        candidates.append(cid)

        # Deterministic ordering: sort the final candidate list rather
        # than relying on dict/list insertion order across index builds,
        # so output is stable regardless of S2/S3 input ordering or
        # Python dict implementation details.
        candidates.sort()

        candidate_map[rec.entity_id] = candidates

    return candidate_map


def _b_record_tokens(rec: NormalizedRecord, min_len: int) -> set:
    """Return the set of usable name tokens for Pass B.

    Primary source is `name_tokens`; falls back to splitting `name_norm`
    / `name_core` only when `name_tokens` yields nothing (robustness for
    hand-built records). Blank tokens and tokens shorter than `min_len`
    are ignored. Never touches country.
    """
    toks: set = set()
    raw = getattr(rec, "name_tokens", None) or []
    for t in raw:
        if t is None:
            continue
        s = t if isinstance(t, str) else str(t)
        s = s.strip()
        if not s or is_blank(s):
            continue
        if len(s) < min_len:
            continue
        toks.add(s)
    if not toks:
        for field in (getattr(rec, "name_norm", None), getattr(rec, "name_core", None)):
            if is_blank(field):
                continue
            for part in str(field).split():
                s = part.strip()
                if s and not is_blank(s) and len(s) >= min_len:
                    toks.add(s)
            if toks:
                break
    return toks


def _pass_b_rare_token(
    s1_norm: List[NormalizedRecord],
    s2_norm: List[NormalizedRecord],
    s3_norm: List[NormalizedRecord],
    config: dict,
) -> Dict[str, List[str]]:
    """Pass B: Rare/IDF Token Blocking.

    Corpus statistics are built over S2+S3 only; document frequency
    counts each token at most once per record. Tokens shorter than
    `pass_b_min_token_len` are ignored. A token is distinctive iff its
    DF is at or below min(`pass_b_max_df_floor`,
    max(2, int(`pass_b_max_df_ratio` * N))). Only distinctive tokens
    are indexed. Each S1 record queries its rarest `pass_b_top_k_tokens`
    indexed tokens via O(1) lookups; per-S1 output is capped at
    `pass_b_max_candidates` (ranked by shared-token count, id
    tie-broken) and returned sorted. Blank-name safe, no country
    filtering, S2/S3 ids only, every S1 key preserved, deterministic.
    """
    cfg = config or {}
    try:
        min_len = int(cfg.get("pass_b_min_token_len", 2))
    except (TypeError, ValueError):
        min_len = 2
    try:
        max_floor = int(cfg.get("pass_b_max_df_floor", 50))
    except (TypeError, ValueError):
        max_floor = 50
    try:
        ratio = float(cfg.get("pass_b_max_df_ratio", 0.01))
    except (TypeError, ValueError):
        ratio = 0.01
    try:
        top_k = int(cfg.get("pass_b_top_k_tokens", 3))
    except (TypeError, ValueError):
        top_k = 3
    try:
        max_cand = int(cfg.get("pass_b_max_candidates", 200))
    except (TypeError, ValueError):
        max_cand = 200
    if min_len < 1:
        min_len = 1
    if top_k < 1:
        top_k = 1

    s2_s3_norm = s2_norm + s3_norm
    n_docs = len(s2_s3_norm)

    candidate_map: Dict[str, List[str]] = {}
    if not s1_norm:
        return candidate_map
    if n_docs == 0:
        for rec in s1_norm:
            candidate_map[rec.entity_id] = []
        return candidate_map

    rel_cap = int(ratio * n_docs) if ratio and ratio > 0 else n_docs
    if rel_cap < 2:
        rel_cap = 2
    threshold = min(max_floor, rel_cap) if max_floor and max_floor > 0 else rel_cap
    if threshold < 1:
        threshold = 1

    from collections import Counter as _Counter

    df: _Counter = _Counter()
    for rec in s2_s3_norm:
        df.update(_b_record_tokens(rec, min_len))

    index: Dict[str, List[str]] = {}
    for rec in s2_s3_norm:
        for tok in _b_record_tokens(rec, min_len):
            if df.get(tok, 0) <= threshold:
                index.setdefault(tok, []).append(rec.entity_id)

    for rec in s1_norm:
        qtoks = [t for t in _b_record_tokens(rec, min_len) if t in index]
        if not qtoks:
            candidate_map[rec.entity_id] = []
            continue
        qtoks.sort(key=lambda t: (df.get(t, 0), t))
        qtoks = qtoks[:top_k]
        counts: Dict[str, int] = {}
        for tok in qtoks:
            for cid in index.get(tok, []):
                counts[cid] = counts.get(cid, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if max_cand and max_cand > 0:
            ranked = ranked[:max_cand]
        out = sorted([cid for cid, _ in ranked])
        candidate_map[rec.entity_id] = out

    return candidate_map


def _c_record_anchors(
    rec: NormalizedRecord,
    use_postal: bool,
    use_numeric: bool,
    use_addr: bool,
    min_num_len: int,
    min_addr_len: int,
) -> set:
    """Return the set of address anchors for Pass C.

    Sources: `postal_tokens`, `numeric_tokens`, `address_tokens`
    (gated by the use_* flags). Blank-address records — detected via
    `is_blank(rec.address_norm)` / missingness information, never via
    the non-existent `address_is_blank` attribute — naturally yield an
    empty set. Short numeric/address tokens are filtered by length.
    Never touches country.
    """
    if is_blank(getattr(rec, "address_norm", None)):
        # Missing/blank address: no anchors (lists are empty anyway,
        # but short-circuit explicitly for safety and clarity).
        # Note: do NOT reference rec.address_is_blank (does not exist).
        pass
    anchors: set = set()
    if use_postal:
        for t in (getattr(rec, "postal_tokens", None) or []):
            if t is None:
                continue
            s = t if isinstance(t, str) else str(t)
            s = s.strip()
            if s and not is_blank(s):
                anchors.add(s)
    if use_numeric:
        for t in (getattr(rec, "numeric_tokens", None) or []):
            if t is None:
                continue
            s = t if isinstance(t, str) else str(t)
            s = s.strip()
            if s and not is_blank(s) and len(s) >= min_num_len:
                anchors.add(s)
    if use_addr:
        for t in (getattr(rec, "address_tokens", None) or []):
            if t is None:
                continue
            s = t if isinstance(t, str) else str(t)
            s = s.strip()
            if s and not is_blank(s) and len(s) >= min_addr_len:
                anchors.add(s)
    return anchors


def _pass_c_numeric_address(
    s1_norm: List[NormalizedRecord],
    s2_norm: List[NormalizedRecord],
    s3_norm: List[NormalizedRecord],
    config: dict,
) -> Dict[str, List[str]]:
    """Pass C: Numeric / Address Anchor Blocking.

    Indexes distinctive postal/numeric/address anchors over S2+S3 only.
    An anchor is distinctive iff its DF is at or below
    min(`pass_c_max_df_floor`, max(2, int(`pass_c_max_df_ratio` * N))),
    suppressing giant buckets from ubiquitous numbers/words. Each S1
    record queries its rarest `pass_c_top_k_anchors` indexed anchors via
    O(1) lookups; a single shared anchor suffices (postal/numeric
    agreement is strong evidence). Per-S1 output is capped at
    `pass_c_max_candidates` (ranked by shared-anchor count, id
    tie-broken) and returned sorted. Blank-address safe, no country
    filtering, S2/S3 ids only, every S1 key preserved, deterministic.
    """
    cfg = config or {}
    use_postal = bool(cfg.get("pass_c_use_postal", True))
    use_numeric = bool(cfg.get("pass_c_use_numeric", True))
    use_addr = bool(cfg.get("pass_c_use_address_tokens", False))
    try:
        min_num_len = int(cfg.get("pass_c_min_numeric_len", 3))
    except (TypeError, ValueError):
        min_num_len = 3
    try:
        min_addr_len = int(cfg.get("pass_c_min_address_token_len", 4))
    except (TypeError, ValueError):
        min_addr_len = 4
    try:
        max_floor = int(cfg.get("pass_c_max_df_floor", 50))
    except (TypeError, ValueError):
        max_floor = 50
    try:
        ratio = float(cfg.get("pass_c_max_df_ratio", 0.005))
    except (TypeError, ValueError):
        ratio = 0.005
    try:
        top_k = int(cfg.get("pass_c_top_k_anchors", 3))
    except (TypeError, ValueError):
        top_k = 3
    try:
        max_cand = int(cfg.get("pass_c_max_candidates", 200))
    except (TypeError, ValueError):
        max_cand = 200
    if min_num_len < 1:
        min_num_len = 1
    if min_addr_len < 1:
        min_addr_len = 1
    if top_k < 1:
        top_k = 1

    s2_s3_norm = s2_norm + s3_norm
    n_docs = len(s2_s3_norm)

    candidate_map: Dict[str, List[str]] = {}
    if not s1_norm:
        return candidate_map
    if n_docs == 0:
        for rec in s1_norm:
            candidate_map[rec.entity_id] = []
        return candidate_map

    rel_cap = int(ratio * n_docs) if ratio and ratio > 0 else n_docs
    if rel_cap < 2:
        rel_cap = 2
    threshold = min(max_floor, rel_cap) if max_floor and max_floor > 0 else rel_cap
    if threshold < 1:
        threshold = 1

    from collections import Counter as _Counter

    df: _Counter = _Counter()
    for rec in s2_s3_norm:
        df.update(
            _c_record_anchors(rec, use_postal, use_numeric, use_addr, min_num_len, min_addr_len)
        )

    index: Dict[str, List[str]] = {}
    for rec in s2_s3_norm:
        for anchor in _c_record_anchors(
            rec, use_postal, use_numeric, use_addr, min_num_len, min_addr_len
        ):
            if df.get(anchor, 0) <= threshold:
                index.setdefault(anchor, []).append(rec.entity_id)

    for rec in s1_norm:
        qanchors = [
            a
            for a in _c_record_anchors(
                rec, use_postal, use_numeric, use_addr, min_num_len, min_addr_len
            )
            if a in index
        ]
        if not qanchors:
            candidate_map[rec.entity_id] = []
            continue
        qanchors.sort(key=lambda a: (df.get(a, 0), a))
        qanchors = qanchors[:top_k]
        counts: Dict[str, int] = {}
        for anchor in qanchors:
            for cid in index.get(anchor, []):
                counts[cid] = counts.get(cid, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if max_cand and max_cand > 0:
            ranked = ranked[:max_cand]
        out = sorted([cid for cid, _ in ranked])
        candidate_map[rec.entity_id] = out

    return candidate_map


def _d_source_string(rec: NormalizedRecord) -> Union[str, None]:
    """Return the Pass D gram source string for a record.

    Uses `name_norm` (full normalized name) when present, falling back
    to `name_core` when `name_norm` is blank. Spaces are removed so that
    grams capture brand-substring evidence rather than ubiquitous
    whitespace grams (which would otherwise create giant buckets and
    false positives on tiny corpora where DF filtering cannot help).
    Returns None when no usable name exists. Never touches country.
    """
    src = rec.name_norm
    if is_blank(src):
        src = rec.name_core
    if is_blank(src):
        return None
    condensed = src.replace(" ", "")
    if len(condensed) < 3:
        return None
    return condensed


def _d_char_grams(s: str, sizes) -> set:
    """Generate the set of character n-grams for a condensed string."""
    grams: set = set()
    n_chars = len(s)
    for n in sizes:
        try:
            n_int = int(n)
        except (TypeError, ValueError):
            continue
        if n_int < 1 or n_int > n_chars:
            continue
        for i in range(n_chars - n_int + 1):
            grams.add(s[i:i + n_int])
    return grams


def _pass_d_char_ngram(
    s1_norm: List[NormalizedRecord],
    s2_norm: List[NormalizedRecord],
    s3_norm: List[NormalizedRecord],
    config: dict,
) -> Dict[str, List[str]]:
    """Pass D: Character 3/4-gram Blocking (typo/noise tolerant).

    Builds an inverted index over S2+S3 keyed by *distinctive* character
    n-grams of the condensed normalized name (spaces removed). A gram is
    distinctive iff its S2+S3 document frequency is at or below an
    effective threshold derived from config (absolute + relative caps),
    which suppresses ubiquitous substrings ("tra", "ing", "ent") that
    would otherwise cause candidate explosion.

    For each S1 record, only its rarest K grams (present in the index)
    are queried; a candidate is emitted iff it shares at least
    `pass_d_min_shared_grams` distinctive grams with the S1 record.
    This is O(1) dict lookups per gram — never an S1 x (S2+S3) scan.

    Guarantees (same as Pass A): entry for every S1 id (possibly []),
    only S2/S3 ids, no duplicates, deterministic sorted order, blank-name
    safe, no country filtering.
    """
    cfg = config or {}
    sizes = cfg.get("pass_d_ngram_sizes", (3, 4))
    # Normalize sizes to a deterministic sorted tuple of ints.
    try:
        sizes = tuple(sorted({int(n) for n in sizes}))
    except TypeError:
        sizes = (3, 4)
    if not sizes:
        sizes = (3, 4)
    max_abs = cfg.get("pass_d_max_document_frequency", 100)
    try:
        max_abs = int(max_abs)
    except (TypeError, ValueError):
        max_abs = 100
    frac = cfg.get("pass_d_max_document_fraction", 0.05)
    try:
        frac = float(frac)
    except (TypeError, ValueError):
        frac = 0.05
    max_grams = cfg.get("pass_d_max_grams_per_record", 30)
    try:
        max_grams = int(max_grams)
    except (TypeError, ValueError):
        max_grams = 30
    min_shared = cfg.get("pass_d_min_shared_grams", 2)
    try:
        min_shared = int(min_shared)
    except (TypeError, ValueError):
        min_shared = 2
    if min_shared < 1:
        min_shared = 1
    if max_grams < 1:
        max_grams = 1

    s2_s3_norm = s2_norm + s3_norm
    n_docs = len(s2_s3_norm)

    candidate_map: Dict[str, List[str]] = {}
    if not s1_norm:
        return candidate_map
    if n_docs == 0:
        for rec in s1_norm:
            candidate_map[rec.entity_id] = []
        return candidate_map

    # Effective DF threshold: min(absolute, max(2, int(frac * N))).
    # max(2, ...) keeps tiny test corpora usable (all grams DF<=2 kept),
    # while the absolute cap bounds buckets on million-row corpora.
    rel_cap = int(frac * n_docs) if frac and frac > 0 else n_docs
    if rel_cap < 2:
        rel_cap = 2
    threshold = min(max_abs, rel_cap) if max_abs and max_abs > 0 else rel_cap
    if threshold < 1:
        threshold = 1

    # Pass 1 (transient sets only): document frequencies. Two-pass design
    # keeps memory at O(DF + index) instead of O(N * grams) retained sets.
    from collections import Counter as _Counter

    df: _Counter = _Counter()
    for rec in s2_s3_norm:
        src = _d_source_string(rec)
        if src is None:
            continue
        df.update(_d_char_grams(src, sizes))

    # Pass 2: inverted index over distinctive grams only.
    index: Dict[str, List[str]] = {}
    for rec in s2_s3_norm:
        src = _d_source_string(rec)
        if src is None:
            continue
        for gram in _d_char_grams(src, sizes):
            if df.get(gram, 0) <= threshold:
                index.setdefault(gram, []).append(rec.entity_id)

    # Retrieval per S1 record.
    for rec in s1_norm:
        src = _d_source_string(rec)
        if src is None:
            candidate_map[rec.entity_id] = []
            continue
        query_grams = [g for g in _d_char_grams(src, sizes) if g in index]
        if not query_grams:
            candidate_map[rec.entity_id] = []
            continue
        # Rarest-first, gram-tie-broken for determinism; cap query fan-out.
        query_grams.sort(key=lambda g: (df.get(g, 0), g))
        query_grams = query_grams[:max_grams]

        counts: Dict[str, int] = {}
        for gram in query_grams:
            for cid in index.get(gram, []):
                counts[cid] = counts.get(cid, 0) + 1

        out = [cid for cid, c in counts.items() if c >= min_shared]
        out.sort()
        candidate_map[rec.entity_id] = out

    return candidate_map


def _e_record_script_key(rec: NormalizedRecord) -> Union[str, None]:
    """Return the deterministic script-routing key for Pass E.

    Reads ONLY `rec.script_flags["name"]` with shape
    {"scripts_present": [...], "primary_script": str|None,
    "is_mixed_script": bool}. Returns the primary script string, or
    None when no usable script was detected (blank/missing names).
    Never touches country; never transliterates.
    """
    flags = getattr(rec, "script_flags", None) or {}
    name_flags = flags.get("name", None) or {}
    primary = name_flags.get("primary_script", None)
    if primary is None:
        return None
    s = primary if isinstance(primary, str) else str(primary)
    s = s.strip()
    if not s or is_blank(s):
        return None
    return s


def _e_record_scripts(rec: NormalizedRecord, mixed_expansion: bool) -> List[str]:
    """Return the deterministic list of query/index scripts for a record.

    Single-script records yield [primary]. Mixed-script records yield
    each script in sorted `scripts_present` when `mixed_expansion` is
    True, else [primary]. Records with no usable script yield [] and
    are never indexed or queried (no giant "missing" bucket).
    """
    flags = getattr(rec, "script_flags", None) or {}
    name_flags = flags.get("name", None) or {}
    primary = _e_record_script_key(rec)
    if primary is None:
        return []
    present = name_flags.get("scripts_present", None) or []
    clean = sorted({str(s).strip() for s in present if s and str(s).strip()})
    is_mixed = bool(name_flags.get("is_mixed_script", False))
    if mixed_expansion and is_mixed and len(clean) > 1:
        return clean
    return [primary]


def _e_record_tokens(rec: NormalizedRecord, min_len: int) -> set:
    """Return usable name tokens for Pass E (same fallback as Pass B)."""
    toks: set = set()
    raw = getattr(rec, "name_tokens", None) or []
    for t in raw:
        if t is None:
            continue
        s = t if isinstance(t, str) else str(t)
        s = s.strip()
        if not s or is_blank(s):
            continue
        if len(s) < min_len:
            continue
        toks.add(s)
    if not toks:
        for field in (getattr(rec, "name_norm", None), getattr(rec, "name_core", None)):
            if is_blank(field):
                continue
            for part in str(field).split():
                s = part.strip()
                if s and not is_blank(s) and len(s) >= min_len:
                    toks.add(s)
            if toks:
                break
    return toks


def _pass_e_script_aware(
    s1_norm: List[NormalizedRecord],
    s2_norm: List[NormalizedRecord],
    s3_norm: List[NormalizedRecord],
    config: dict,
) -> Dict[str, List[str]]:
    """Pass E: Script-Aware Blocking.

    Composite inverted index over S2+S3 keyed by
    (script_key, token): script routing comes ONLY from
    `rec.script_flags["name"]`, matching evidence ONLY from
    `rec.name_tokens` (fallback `name_norm`/`name_core` split, Pass-B
    style). Mixed-script S2+S3 records are indexed under each of their
    scripts when `pass_e_use_mixed_expansion` is True. Document
    frequency is counted once per record per (script, token); only
    pairs with DF at or below min(`pass_e_max_df_floor`,
    max(2, int(`pass_e_max_df_ratio` * N))) enter the index, and any
    bucket larger than `pass_e_max_bucket_size` is never used for
    retrieval (Latin mega-bucket guard). Each S1 record queries its
    rarest `pass_e_top_k_tokens` indexed pairs, ranks candidates by
    (-shared, id), keeps `pass_e_max_candidates`, returns sorted.
    Blank/missing-script records yield [] (never a "missing" bucket).
    No country reads, no transliteration, S2/S3 ids only, every S1 key
    preserved, deterministic.
    """
    cfg = config or {}
    try:
        min_len = int(cfg.get("pass_e_min_token_len", 2))
    except (TypeError, ValueError):
        min_len = 2
    try:
        max_floor = int(cfg.get("pass_e_max_df_floor", 50))
    except (TypeError, ValueError):
        max_floor = 50
    try:
        ratio = float(cfg.get("pass_e_max_df_ratio", 0.01))
    except (TypeError, ValueError):
        ratio = 0.01
    try:
        top_k = int(cfg.get("pass_e_top_k_tokens", 3))
    except (TypeError, ValueError):
        top_k = 3
    try:
        max_cand = int(cfg.get("pass_e_max_candidates", 200))
    except (TypeError, ValueError):
        max_cand = 200
    try:
        max_bucket = int(cfg.get("pass_e_max_bucket_size", 500))
    except (TypeError, ValueError):
        max_bucket = 500
    mixed_expansion = bool(cfg.get("pass_e_use_mixed_expansion", True))
    if min_len < 1:
        min_len = 1
    if top_k < 1:
        top_k = 1

    s2_s3_norm = s2_norm + s3_norm
    n_docs = len(s2_s3_norm)

    candidate_map: Dict[str, List[str]] = {}
    if not s1_norm:
        return candidate_map
    if n_docs == 0:
        for rec in s1_norm:
            candidate_map[rec.entity_id] = []
        return candidate_map

    rel_cap = int(ratio * n_docs) if ratio and ratio > 0 else n_docs
    if rel_cap < 2:
        rel_cap = 2
    threshold = min(max_floor, rel_cap) if max_floor and max_floor > 0 else rel_cap
    if threshold < 1:
        threshold = 1

    from collections import Counter as _Counter

    df: _Counter = _Counter()
    for rec in s2_s3_norm:
        scripts = _e_record_scripts(rec, mixed_expansion)
        if not scripts:
            continue
        toks = _e_record_tokens(rec, min_len)
        if not toks:
            continue
        for script in scripts:
            for tok in toks:
                df[(script, tok)] += 1

    index: Dict[tuple, List[str]] = {}
    for rec in s2_s3_norm:
        scripts = _e_record_scripts(rec, mixed_expansion)
        if not scripts:
            continue
        toks = _e_record_tokens(rec, min_len)
        if not toks:
            continue
        for script in scripts:
            for tok in toks:
                key = (script, tok)
                if df.get(key, 0) <= threshold:
                    index.setdefault(key, []).append(rec.entity_id)

    for rec in s1_norm:
        scripts = _e_record_scripts(rec, mixed_expansion)
        toks = _e_record_tokens(rec, min_len)
        if not scripts or not toks:
            candidate_map[rec.entity_id] = []
            continue
        pairs = [
            (script, tok)
            for script in scripts
            for tok in toks
            if (script, tok) in index
            and len(index.get((script, tok), [])) <= max_bucket
        ]
        if not pairs:
            candidate_map[rec.entity_id] = []
            continue
        pairs.sort(key=lambda p: (df.get(p, 0), p[0], p[1]))
        pairs = pairs[:top_k]
        counts: Dict[str, int] = {}
        for key in pairs:
            for cid in index.get(key, []):
                counts[cid] = counts.get(cid, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if max_cand and max_cand > 0:
            ranked = ranked[:max_cand]
        candidate_map[rec.entity_id] = sorted([cid for cid, _ in ranked])

    return candidate_map


def _union_candidate_maps(*maps: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Union any number of candidate maps.

    Keys are the union of all S1 ids; values are the sorted union of
    candidate ids (deduped). Deterministic. Assumes values contain only
    S2/S3 ids (as each pass guarantees); this function does not add or
    filter ids beyond unioning.
    """
    union: Dict[str, List[str]] = {}
    all_keys: set = set()
    for m in maps:
        if m:
            all_keys.update(m.keys())
    for key in all_keys:
        seen: set = set()
        for m in maps:
            if m and key in m:
                for cid in m[key]:
                    seen.add(cid)
        union[key] = sorted(seen)
    return union


def generate_candidates(
    s1_records: Sequence[RawOrNormalized],
    s2_records: Sequence[RawOrNormalized],
    s3_records: Sequence[RawOrNormalized],
    config: dict,
) -> Dict[str, List[str]]:
    """Top-level blocking entry point.

    Normalizes raw input (if needed) via Nidhi's normalize_dataframe_rows(),
    then runs the implemented blocking passes (A + B + C + D + E, unioned).

    Guarantees:
      - Every S1 record has an entry in the returned dict (possibly []).
      - Values contain only S2/S3 entity_ids.
      - No duplicate ids within a value.
      - Deterministic ordering.
    """
    s1_norm = _ensure_normalized(s1_records)
    s2_norm = _ensure_normalized(s2_records)
    s3_norm = _ensure_normalized(s3_records)

    cfg = config or {}

    candidate_map_a = _pass_a_exact_core_name(s1_norm, s2_norm, s3_norm, cfg)

    maps = [candidate_map_a]
    if cfg.get("pass_b_enabled", True):
        maps.append(_pass_b_rare_token(s1_norm, s2_norm, s3_norm, cfg))
    if cfg.get("pass_c_enabled", True):
        maps.append(_pass_c_numeric_address(s1_norm, s2_norm, s3_norm, cfg))
    if cfg.get("pass_d_enabled", True):
        maps.append(_pass_d_char_ngram(s1_norm, s2_norm, s3_norm, cfg))
    if cfg.get("pass_e_enabled", True):
        maps.append(_pass_e_script_aware(s1_norm, s2_norm, s3_norm, cfg))
    candidate_map = _union_candidate_maps(*maps)

    return candidate_map
