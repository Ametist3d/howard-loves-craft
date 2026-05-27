from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# This module intentionally has no LLM calls.
# It is a small, deterministic reranker for rulebook retrieval.

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "if", "player", "wants",
    "want", "what", "which", "when", "where", "why", "how", "with", "into", "from",
    "that", "this", "these", "those", "then", "than", "can", "could", "would", "should",
    "make", "does", "using", "use", "about", "before", "after", "there", "their",
}


@dataclass(slots=True)
class RetrievalConfig:
    """Config for deterministic post-retrieval reranking.

    Chroma/LangChain often returns distance-like scores where lower is better.
    `dense_distance_is_lower_better=True` keeps that assumption explicit.
    """

    k_per_query: int = 8
    top_n: int = 4
    max_query_variants: int = 6
    max_candidates: int = 64

    dense_weight: float = 1.0
    text_overlap_weight: float = 0.35
    header_overlap_weight: float = 0.45
    phrase_weight: float = 0.20
    length_weight: float = 0.08

    dense_distance_is_lower_better: bool = True

    source_priority: Dict[str, float] = field(default_factory=dict)
    content_type_priority: Dict[str, float] = field(default_factory=dict)
    negative_source_patterns: List[str] = field(default_factory=list)
    negative_header_patterns: List[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedItem:
    doc: Any
    vector_score: float
    final_score: float
    reasons: List[str]


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁіІїЇєЄґҐ]{3,}", normalize(text))
    return [t for t in tokens if t not in STOPWORDS]


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        clean = re.sub(r"\s+", " ", str(item or "")).strip()
        key = normalize(clean)
        if key and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def build_query_variants(query: str, *, max_variants: int = 6) -> List[str]:
    """Build cheap lexical variants for the same rules query.

    No domain-specific hardcoding here. The engine should pass a good mechanics query.
    """
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not query:
        return []

    tokens = tokenize(query)
    variants: List[str] = [query]

    if tokens:
        variants.append(" ".join(tokens))

    if len(tokens) >= 2:
        variants.extend(" ".join(tokens[i:i + 2]) for i in range(min(len(tokens) - 1, 4)))

    if len(tokens) >= 3:
        variants.extend(" ".join(tokens[i:i + 3]) for i in range(min(len(tokens) - 2, 3)))

    return unique_keep_order(variants)[:max(1, int(max_variants or 1))]


def metadata_blob(meta: Dict[str, Any]) -> str:
    keys = (
        "Header_1", "Header_2", "Header_3", "Header_4",
        "title_en", "title_original", "display_name", "source", "content_type",
    )
    return " | ".join(str(meta.get(k, "") or "") for k in keys if str(meta.get(k, "") or "").strip())


def token_overlap(query_tokens: Sequence[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    return sum(1 for t in query_tokens if t in text_tokens) / max(1, len(query_tokens))


def phrase_bonus(query_tokens: Sequence[str], text: str) -> float:
    haystack = normalize(text)
    bonus = 0.0

    for size, weight in ((4, 1.15), (3, 1.0), (2, 0.6)):
        if len(query_tokens) < size:
            continue
        for i in range(len(query_tokens) - size + 1):
            phrase = " ".join(query_tokens[i:i + size])
            if phrase and phrase in haystack:
                bonus = max(bonus, weight)

    return bonus


def chunk_length_bonus(text: str) -> float:
    n = len(str(text or "").strip())
    if 180 <= n <= 1400:
        return 1.0
    if 100 <= n <= 2200:
        return 0.5
    return 0.0


def _priority_lookup(mapping: Dict[str, float], value: str) -> float:
    wanted = normalize(value)
    if not wanted:
        return 0.0

    for key, score in (mapping or {}).items():
        key_norm = normalize(str(key))
        if not key_norm:
            continue
        if wanted == key_norm or key_norm in wanted or wanted in key_norm:
            return float(score or 0.0)
    return 0.0


def source_adjustment(meta: Dict[str, Any], cfg: RetrievalConfig) -> Tuple[float, List[str]]:
    bonus = 0.0
    reasons: List[str] = []

    source = str(meta.get("source", "") or "")
    content_type = str(meta.get("content_type", "") or "")
    headers = " ".join(
        str(meta.get(k, "") or "")
        for k in ("Header_1", "Header_2", "Header_3", "Header_4")
    )

    source_bonus = _priority_lookup(cfg.source_priority, source)
    if source_bonus:
        bonus += source_bonus
        reasons.append(f"source_priority={source_bonus:+.2f}")

    content_bonus = _priority_lookup(cfg.content_type_priority, content_type)
    if content_bonus:
        bonus += content_bonus
        reasons.append(f"content_type_priority={content_bonus:+.2f}")

    source_norm = normalize(source)
    headers_norm = normalize(headers)

    for pattern in cfg.negative_source_patterns:
        if re.search(pattern, source_norm, flags=re.I):
            bonus -= 0.10
            reasons.append(f"negative_source_pattern:{pattern}")
            break

    for pattern in cfg.negative_header_patterns:
        if re.search(pattern, headers_norm, flags=re.I):
            bonus -= 0.08
            reasons.append(f"negative_header_pattern:{pattern}")
            break

    return bonus, reasons


def dedupe_key(doc: Any) -> Tuple[str, str, str, str]:
    meta = getattr(doc, "metadata", {}) or {}
    page = str(getattr(doc, "page_content", "") or "")
    return (
        normalize(str(meta.get("source", "") or "")),
        normalize(str(meta.get("Header_1", "") or "")),
        normalize(str(meta.get("Header_2", "") or "")),
        normalize(page[:260]),
    )


def _dense_component(vector_score: float, cfg: RetrievalConfig) -> float:
    score = float(vector_score)
    if not math.isfinite(score):
        score = 999.0 if cfg.dense_distance_is_lower_better else -999.0
    return -score * cfg.dense_weight if cfg.dense_distance_is_lower_better else score * cfg.dense_weight


def rerank_candidates(
    query: str,
    candidates: List[Tuple[Any, float]],
    cfg: RetrievalConfig,
) -> List[RetrievedItem]:
    query_tokens = tokenize(query)
    seen = set()
    out: List[RetrievedItem] = []

    for doc, vec_score in candidates[: max(1, int(cfg.max_candidates or 1))]:
        key = dedupe_key(doc)
        if key in seen:
            continue
        seen.add(key)

        meta = getattr(doc, "metadata", {}) or {}
        text = str(getattr(doc, "page_content", "") or "")
        meta_text = metadata_blob(meta)
        combined = f"{text}\n{meta_text}"

        final = _dense_component(float(vec_score), cfg)
        reasons = [f"vector={float(vec_score):.4f}"]

        text_score = token_overlap(query_tokens, text)
        if text_score:
            final += text_score * cfg.text_overlap_weight
            reasons.append(f"text_overlap={text_score:.3f}")

        header_score = token_overlap(query_tokens, meta_text)
        if header_score:
            final += header_score * cfg.header_overlap_weight
            reasons.append(f"header_overlap={header_score:.3f}")

        phrase = phrase_bonus(query_tokens, combined)
        if phrase:
            final += phrase * cfg.phrase_weight
            reasons.append(f"phrase_bonus={phrase:.3f}")

        length = chunk_length_bonus(text)
        if length:
            final += length * cfg.length_weight
            reasons.append(f"length_bonus={length:.2f}")

        adjustment, adjustment_reasons = source_adjustment(meta, cfg)
        final += adjustment
        reasons.extend(adjustment_reasons)

        out.append(RetrievedItem(doc=doc, vector_score=float(vec_score), final_score=final, reasons=reasons))

    out.sort(key=lambda item: item.final_score, reverse=True)
    return out[: max(1, int(cfg.top_n or 1))]


def retrieve_with_rerank(
    db: Any,
    query: str,
    cfg: RetrievalConfig | None = None,
) -> List[RetrievedItem]:
    """Retrieve from a LangChain/Chroma-like vector store and rerank locally.

    Expected db method: similarity_search_with_score(query, k=n).
    Fallback method: similarity_search(query, k=n), treated as neutral score 0.0.
    """
    cfg = cfg or RetrievalConfig()
    variants = build_query_variants(query, max_variants=cfg.max_query_variants)
    if not db or not variants:
        return []

    raw_hits: List[Tuple[Any, float]] = []

    for variant in variants:
        try:
            if hasattr(db, "similarity_search_with_score"):
                raw_hits.extend(db.similarity_search_with_score(variant, k=cfg.k_per_query))
            elif hasattr(db, "similarity_search"):
                docs = db.similarity_search(variant, k=cfg.k_per_query)
                raw_hits.extend((doc, 0.0) for doc in docs)
        except Exception:
            # Keep retrieval resilient. Caller can still fallback if result is empty.
            continue

    if not raw_hits:
        return []

    return rerank_candidates(query=query, candidates=raw_hits, cfg=cfg)


__all__ = [
    "RetrievalConfig",
    "RetrievedItem",
    "build_query_variants",
    "retrieve_with_rerank",
    "rerank_candidates",
]
