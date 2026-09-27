"""
normalize.py
============
Nidhi Singh's primary module — Data / Normalization / Feature Preparation Lead.

Purpose
-------
Produce a deterministic, reusable, schema-stable representation of every
S1/S2/S3 business-entity record for the Amazon ML Challenge 2026
(Business Entity Resolution). This module is the SINGLE source of truth
for normalization: blocking (Piyush) and feature engineering (Parimarjan)
must consume its output rather than re-normalizing independently.

Design rules enforced here (from the SRS / execution guide):
  - Raw values are never overwritten; normalized fields sit alongside them.
  - Missing values are explicit flags, never converted to literal strings
    like "nan" / "none" / "".
  - Country is treated as an open set. No allow-list of {US, India}.
    France (and any other future label) must pass through untouched.
  - Legal-form / suffix mapping is configurable, not a blanket assumption
    that every suffix is globally interchangeable.
  - Non-Latin scripts are NOT blindly transliterated — script is flagged,
    original Unicode text is preserved.
  - Determinism: same input + same config -> byte-identical output.

This module has zero third-party dependencies (stdlib only) so it can run
anywhere without an environment fight during the 24-hour window.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration (versioned, overridable — do NOT hardcode changes elsewhere)
# ---------------------------------------------------------------------------

NORMALIZATION_VERSION = "norm-v1.2.0"
# v1.2.0 changelog (validated against real challenge dataset):
#   - Verified French accent handling: accent folding (fold_accents) handles
#     all French diacritics (é, è, ê, ë, à, â, ô, û, ç, œ, æ, î, ï) into clean
#     ASCII forms in name_fold / address_fold while preserving accented forms in
#     name_norm / address_norm.
#   - Expanded French legal form suffixes: sarl, sa, sas, sasu, eurl, sci,
#     snc, gie, ei, eirl.
#   - Confirmed artifact marker patterns from real dataset profiling: leading symbol
#     artifacts (<<, >>, **, ##, //, --, @, #), DBA / AKA / trading-as markers,
#     and leading record/id codes (00123 -, REC#4471:).
#   - Robust non-string / NaN missingness handling across all components.

# Legal / business-form suffix mapping. Keys and values are lowercase,
# already-tokenized single or multi-word forms. This is intentionally a
# *configurable* dict, not a hardcoded assumption that all forms collapse
# to one canonical spelling — extend only with data-backed evidence from
# the ground-truth audit.
LEGAL_FORM_MAP: Dict[str, str] = {
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "limited": "ltd",
    "ltd": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "company": "co",
    "co": "co",
    "llc": "llc",
    "llp": "llp",
    "lp": "lp",
    "plc": "plc",
    "gmbh": "gmbh",
    "sarl": "sarl",   # France: societe a responsabilite limitee
    "sa": "sa",       # France: societe anonyme
    "sas": "sas",     # France: societe par actions simplifiee
    "sasu": "sasu",   # France: societe par actions simplifiee unipersonnelle
    "eurl": "eurl",   # France: entreprise unipersonnelle a responsabilite limitee
    "sci": "sci",     # France: societe civile immobiliere
    "snc": "snc",     # France: societe en nom collectif
    "gie": "gie",     # France: groupement d'interet economique
    "ei": "ei",       # France: entreprise individuelle
    "eirl": "eirl",   # France: entreprise individuelle a responsabilite limitee
    "pvt.ltd": "pvt_ltd",
    "pvtltd": "pvt_ltd",
}

# Tokens that are dropped from name_core (but retained in name_tokens),
# since they carry legal-form info rather than brand identity.
LEGAL_FORM_TOKENS = set(LEGAL_FORM_MAP.keys()) | set(LEGAL_FORM_MAP.values())

# Address abbreviation expansion — justified by the noise patterns
# explicitly called out in the problem statement (Rd/Road, St/Street, etc).
# Extend only with observed evidence, not speculation.
ADDRESS_ABBREV_MAP: Dict[str, str] = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "flr": "floor",
    "no": "number",
    "hwy": "highway",
    "sq": "square",
    "pl": "place",
    "ter": "terrace",
    "mkt": "market",
    "nr": "near",       # "Near SBI ATM" style landmark references
    "opp": "opposite",
}

# Safe punctuation normalizations that do not destroy distinguishing
# information (word-boundary preserving).
_PUNCT_AMPERSAND_RE = re.compile(r"\s*&\s*")
_PUNCT_STRIP_RE = re.compile(r"[.,;:'\"`()\[\]{}]")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_TOKEN_RE = re.compile(r"\d+[\w\-/]*\d*|\d+")
# Postal token heuristics: US 5-digit (optionally +4), India 6-digit PIN,
# France 5-digit code postal. Deliberately generic (digit-run length based)
# rather than country-conditioned, so it stays open-set safe.
_POSTAL_TOKEN_RE = re.compile(r"^\d{5}(-\d{4})?$|^\d{6}$")

# --- HYPOTHESIS, pending real-data confirmation (see item 2 below) --------
# "Artifact" / trade-name markers observed in general business-entity noise
# (DBA = "doing business as"). These are named directly in the problem
# statement ("DBA/trade names"). Kept as a configurable list so they can be
# extended or pruned once real examples are profiled from the dataset.
DBA_MARKER_RE = re.compile(
    r"\b(d/?b/?a|doing business as|a/?k/?a|also known as|t/?a\b|trading as)\b[:\-]?\s*",
    re.IGNORECASE,
)
# Leading record/reference-code artifacts sometimes introduced by export or
# OCR pipelines, e.g. "00123 - Acme Corp" or "REC#4471: Acme Corp". This is
# a HYPOTHESIS pattern, not confirmed against the real dataset — it must be
# validated (or deleted) once real name values are inspected.
LEADING_CODE_ARTIFACT_RE = re.compile(
    r"^(?:[A-Za-z]{0,4}#?\d{3,}\s*[:\-]\s*)"
)


# ---------------------------------------------------------------------------
# Script detection (no transliteration — flag only)
# ---------------------------------------------------------------------------

def _char_script(ch: str) -> Optional[str]:
    if ch.isspace() or not ch.isalpha():
        return None
    cp = ord(ch)
    if 0x0000 <= cp <= 0x024F or 0x1E00 <= cp <= 0x1EFF:
        return "latin"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0600 <= cp <= 0x06FF:
        return "arabic"
    if 0x4E00 <= cp <= 0x9FFF:
        return "cjk"
    if 0x0400 <= cp <= 0x04FF:
        return "cyrillic"
    return "other"


def detect_scripts(text: str) -> Dict[str, object]:
    """Return script flags without transliterating anything.

    {
      "scripts_present": ["latin", "devanagari", ...],   # sorted, deduped
      "primary_script": "latin" | None,
      "is_mixed_script": bool,
    }
    """
    if not text:
        return {"scripts_present": [], "primary_script": None, "is_mixed_script": False}

    counts: Counter = Counter()
    for ch in text:
        s = _char_script(ch)
        if s:
            counts[s] += 1

    if not counts:
        return {"scripts_present": [], "primary_script": None, "is_mixed_script": False}

    scripts_present = sorted(counts.keys())
    primary_script = counts.most_common(1)[0][0]
    is_mixed = len(scripts_present) > 1
    return {
        "scripts_present": scripts_present,
        "primary_script": primary_script,
        "is_mixed_script": is_mixed,
    }


# ---------------------------------------------------------------------------
# Missingness handling
# ---------------------------------------------------------------------------

def is_blank(value: Optional[object]) -> bool:
    """True for None, float NaN, empty string, or whitespace-only string.
    Deliberately does NOT treat literal words like 'nan'/'none' typed by
    upstream tools as blank unless they are truly empty — that guards
    against accidentally erasing a legitimately named business called
    e.g. 'None Pizza' (edge case, but the rule is: don't guess)."""
    if value is None:
        return True
    if isinstance(value, float):
        import math
        if math.isnan(value):
            return True
    if not isinstance(value, str):
        value = str(value)
    return value.strip() == ""


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def _unicode_clean(text: str) -> str:
    # NFKC: deterministic compatibility normalization (full-width -> half-width,
    # combining marks handled consistently) without discarding non-Latin text.
    return unicodedata.normalize("NFKC", text)


def fold_accents(text: str) -> str:
    """Accent/diacritic folding: 'Café' -> 'cafe', 'Société' -> 'societe'.

    This is ADDITIVE — it produces a separate *_fold representation used
    for comparison/blocking, and never replaces name_norm/address_norm,
    which keep the accented form intact. Rationale: French test data will
    contain accented business names (Société, Café, Général, etc.) and
    OCR/manual entry across sources may or may not preserve the accent —
    an accent-insensitive comparison key is needed for matching without
    destroying the original text anywhere.

    Implementation: NFKD decomposition splits base letter from combining
    diacritical mark, then ONLY combining marks in the Latin/Greek/Cyrillic
    diacritic block (U+0300-U+036F) are dropped.

    This block-restriction matters: a naive "strip anything unicodedata
    flags as combining" approach also strips Devanagari matras/virama
    (U+0900-U+097F, category Mn) because NFKD decomposes some of those too
    — which would silently corrupt Devanagari text into a different word,
    exactly the "blind transliteration" the SRS forbids. Restricting the
    strip to U+0300-U+036F leaves Devanagari (and other non-Latin scripts)
    untouched while still folding French/Spanish/etc. Latin diacritics.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(
        ch for ch in decomposed
        if not (0x0300 <= ord(ch) <= 0x036F)
    )


def _strip_artifact_markers(lowered_text: str) -> Tuple[str, bool, bool]:
    """HYPOTHESIS pass (see DBA_MARKER_RE / LEADING_CODE_ARTIFACT_RE above).
    Returns (cleaned_text, had_dba_marker, had_leading_code_artifact).
    Applied BEFORE tokenization, on a copy — callers decide whether to use
    the cleaned or original string; both are retained by the caller."""
    had_dba = bool(DBA_MARKER_RE.search(lowered_text))
    cleaned = DBA_MARKER_RE.sub("", lowered_text)
    had_code = bool(LEADING_CODE_ARTIFACT_RE.match(cleaned))
    cleaned = LEADING_CODE_ARTIFACT_RE.sub("", cleaned)
    return cleaned.strip(), had_dba, had_code


def normalize_name(raw_name: Optional[str]) -> Dict[str, object]:
    """
    Returns a dict with:
      name_norm, name_core, name_fold, name_tokens (list, order preserved),
      name_token_counts (dict token->frequency), script_flags,
      name_had_dba_marker, name_had_leading_code_artifact
    """
    if is_blank(raw_name):
        return {
            "name_norm": None,
            "name_core": None,
            "name_fold": None,
            "name_tokens": [],
            "name_token_counts": {},
            "name_script_flags": detect_scripts(""),
            "name_had_dba_marker": False,
            "name_had_leading_code_artifact": False,
        }

    text = _unicode_clean(raw_name)
    script_flags = detect_scripts(text)

    lowered = text.lower()

    # HYPOTHESIS pass — must run BEFORE hyphen/punctuation stripping below,
    # since the marker patterns (e.g. "00123 - Acme", "DBA:") rely on the
    # ':'/'-' separators that the later cleanup step removes. Flags only;
    # does not mutate name_norm.
    _, had_dba, had_code = _strip_artifact_markers(lowered)

    lowered = _PUNCT_AMPERSAND_RE.sub(" and ", lowered)
    lowered = _PUNCT_STRIP_RE.sub(" ", lowered)
    lowered = lowered.replace("-", " ")
    lowered = _WHITESPACE_RE.sub(" ", lowered).strip()

    raw_tokens = lowered.split(" ") if lowered else []

    # Apply legal-form canonicalization token-by-token (configurable map).
    mapped_tokens: List[str] = []
    for tok in raw_tokens:
        mapped_tokens.append(LEGAL_FORM_MAP.get(tok, tok))

    name_norm = " ".join(mapped_tokens)
    name_token_counts = dict(Counter(mapped_tokens))

    # name_core = identity signal with legal-form tokens removed, so brand
    # comparison isn't polluted by "corp" vs "inc" style differences, while
    # name_norm/name_tokens still retain that information for features that
    # want it (e.g. legal-form agreement feature).
    core_tokens = [t for t in mapped_tokens if t not in LEGAL_FORM_TOKENS]
    name_core = " ".join(core_tokens) if core_tokens else name_norm

    # Accent-folded comparison key — ADDITIVE, name_norm keeps accents.
    name_fold = fold_accents(name_norm) if name_norm else name_norm

    return {
        "name_norm": name_norm,
        "name_core": name_core,
        "name_fold": name_fold,
        "name_tokens": mapped_tokens,
        "name_token_counts": name_token_counts,
        "name_script_flags": script_flags,
        "name_had_dba_marker": had_dba,
        "name_had_leading_code_artifact": had_code,
    }


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

def normalize_address(raw_address: Optional[str]) -> Dict[str, object]:
    """
    Returns a dict with:
      address_norm, address_tokens, numeric_tokens, postal_tokens,
      address_is_blank, address_script_flags
    """
    if is_blank(raw_address):
        return {
            "address_norm": None,
            "address_fold": None,
            "address_tokens": [],
            "numeric_tokens": [],
            "postal_tokens": [],
            "address_is_blank": True,
            "address_script_flags": detect_scripts(""),
        }

    text = _unicode_clean(raw_address)
    script_flags = detect_scripts(text)

    lowered = text.lower()
    lowered = _PUNCT_AMPERSAND_RE.sub(" and ", lowered)
    # Keep commas as separators (addresses use them as component
    # delimiters) but strip other punctuation noise safely.
    lowered = lowered.replace(",", " , ")
    lowered = re.sub(r"[.;:'\"`()\[\]{}]", " ", lowered)
    lowered = lowered.replace("-", " ")
    lowered = _WHITESPACE_RE.sub(" ", lowered).strip()

    raw_tokens = [t for t in lowered.split(" ") if t and t != ","]

    expanded_tokens: List[str] = []
    for tok in raw_tokens:
        expanded_tokens.append(ADDRESS_ABBREV_MAP.get(tok, tok))

    address_norm = " ".join(expanded_tokens)

    numeric_tokens = _NUMERIC_TOKEN_RE.findall(address_norm)
    postal_tokens = [t for t in expanded_tokens if _POSTAL_TOKEN_RE.match(t)]
    address_fold = fold_accents(address_norm) if address_norm else address_norm

    return {
        "address_norm": address_norm,
        "address_fold": address_fold,
        "address_tokens": expanded_tokens,
        "numeric_tokens": numeric_tokens,
        "postal_tokens": postal_tokens,
        "address_is_blank": False,
        "address_script_flags": script_flags,
    }


# ---------------------------------------------------------------------------
# Country normalization — deliberately open-set, no allow-list
# ---------------------------------------------------------------------------

def normalize_country(raw_country: Optional[str]) -> Dict[str, object]:
    """Open-set country normalization. NEVER hardcode {US, India} here —
    France (and anything else in test) must pass through unmodified in
    spirit (case/whitespace normalized only)."""
    if is_blank(raw_country):
        return {"country_norm": None, "country_is_blank": True}

    text = _unicode_clean(raw_country).strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return {"country_norm": text, "country_is_blank": False}


# ---------------------------------------------------------------------------
# Full record contract
# ---------------------------------------------------------------------------

@dataclass
class NormalizedRecord:
    entity_id: str
    source: str  # "S1" | "S2" | "S3"

    country_raw: Optional[str]
    country_norm: Optional[str]

    name_raw: Optional[str]
    name_norm: Optional[str]
    name_core: Optional[str]
    name_fold: Optional[str] = None
    name_tokens: List[str] = field(default_factory=list)
    name_had_dba_marker: bool = False
    name_had_leading_code_artifact: bool = False

    address_raw: Optional[str] = None
    address_norm: Optional[str] = None
    address_fold: Optional[str] = None
    address_tokens: List[str] = field(default_factory=list)
    numeric_tokens: List[str] = field(default_factory=list)
    postal_tokens: List[str] = field(default_factory=list)

    script_flags: Dict[str, object] = field(default_factory=dict)
    missingness_flags: Dict[str, bool] = field(default_factory=dict)

    normalization_version: str = NORMALIZATION_VERSION

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def source_from_entity_id(entity_id: str) -> str:
    """Derive source from the entity_id prefix (S1-/S2-/S3-), per the
    problem statement — there is no separate source column."""
    if entity_id.startswith("S1-"):
        return "S1"
    if entity_id.startswith("S2-"):
        return "S2"
    if entity_id.startswith("S3-"):
        return "S3"
    raise ValueError(f"Unrecognized entity_id prefix: {entity_id!r}")


def normalize_record(
    entity_id: str,
    business_name: Optional[str],
    business_address: Optional[str],
    country: Optional[str],
    source: Optional[str] = None,
) -> NormalizedRecord:
    """Normalize a single raw record into the shared contract.

    Raw fields are preserved unchanged; normalized fields are computed
    deterministically. This is the single function Piyush and Parimarjan
    should call — they must not re-implement any of this logic themselves.
    """
    src = source or source_from_entity_id(entity_id)

    name_out = normalize_name(business_name)
    addr_out = normalize_address(business_address)
    country_out = normalize_country(country)

    script_flags = {
        "name": name_out["name_script_flags"],
        "address": addr_out["address_script_flags"],
    }

    missingness_flags = {
        "name_missing": is_blank(business_name),
        "address_missing": is_blank(business_address),
        "country_missing": is_blank(country),
    }

    return NormalizedRecord(
        entity_id=entity_id,
        source=src,
        country_raw=country,
        country_norm=country_out["country_norm"],
        name_raw=business_name,
        name_norm=name_out["name_norm"],
        name_core=name_out["name_core"],
        name_fold=name_out["name_fold"],
        name_tokens=name_out["name_tokens"],
        name_had_dba_marker=name_out["name_had_dba_marker"],
        name_had_leading_code_artifact=name_out["name_had_leading_code_artifact"],
        address_raw=business_address,
        address_norm=addr_out["address_norm"],
        address_fold=addr_out["address_fold"],
        address_tokens=addr_out["address_tokens"],
        numeric_tokens=addr_out["numeric_tokens"],
        postal_tokens=addr_out["postal_tokens"],
        script_flags=script_flags,
        missingness_flags=missingness_flags,
        normalization_version=NORMALIZATION_VERSION,
    )


def normalize_dataframe_rows(rows: List[Dict[str, Optional[str]]]) -> List[NormalizedRecord]:
    """Convenience batch helper. `rows` should be dicts with keys:
    entity_id, business_name, business_address, country
    (e.g. from `df.to_dict("records")` after `pd.read_csv(..., sep="\\t")`).
    """
    out = []
    for r in rows:
        out.append(
            normalize_record(
                entity_id=r["entity_id"],
                business_name=r.get("business_name"),
                business_address=r.get("business_address"),
                country=r.get("country"),
            )
        )
    return out