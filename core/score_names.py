from __future__ import annotations

import re
from typing import List


VARIANCE_HETEROGENEITY = "Variance Heterogeneity"
VARIANCE_HETEROGENEITY_STATS = "Variance Heterogeneity Stats"
CHANGE_EFFECT_SCORES = "Change Effect Scores"
CHANGE_EFFECT_DETAIL = "Change Effect Detail"
INTRA_OMICS_CORRELATION = "Intra-omics Correlation"
CROSS_OMICS_CORRELATION = "Cross-omics Correlation"
POSITIVE_LINKS = "Positive Links"
NEGATIVE_LINKS = "Negative Links"
LINK_DETAILS = "Link Details"

DYNAMIC_BASE_SCORE_NAMES: List[str] = [
    VARIANCE_HETEROGENEITY,
    POSITIVE_LINKS,
    NEGATIVE_LINKS,
]

STRUCTURAL_SCORE_NAMES: List[str] = [
    CROSS_OMICS_CORRELATION,
    INTRA_OMICS_CORRELATION,
]


def build_change_effect_score_name(left_label: str, right_label: str) -> str:
    left = str(left_label).strip()
    right = str(right_label).strip()
    return f"{left} vs. {right} Change Effect"


def is_change_effect_score_name(score_name: str) -> bool:
    return str(score_name).strip().endswith(" Change Effect")


def safe_score_filename(score_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(score_name).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "score"
