# pylint: disable=import-error
from utils.rules_retrieval_patch import RetrievalConfig

RULES_RETRIEVAL_CFG = RetrievalConfig(
    k_per_query=8,
    top_n=4,
    max_query_variants=6,
    max_candidates=64,

    dense_weight=1.0,
    text_overlap_weight=0.35,
    header_overlap_weight=0.45,
    phrase_weight=0.20,
    length_weight=0.08,

    # Chroma similarity_search_with_score usually returns distance-like scores.
    dense_distance_is_lower_better=True,

    source_priority={
        "docero.tips_call-of-cthulhu-7th-ed-keeper-rulebook.md": 0.10,
        "call_of_cthulhu_7th_edition_quick-start_rules_rus.md": 0.04,
    },
    content_type_priority={
        "rulebook": 0.08,
    },
    negative_source_patterns=[
        r"scenario",
        r"adventure",
        r"starter",
    ],
    negative_header_patterns=[
        r"corbitt",
        r"turner",
        r"cabin",
        r"casket",
        r"keeper note",
        r"example",
    ],
)
