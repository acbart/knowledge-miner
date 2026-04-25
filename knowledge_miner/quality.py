"""Quality scoring for KC maps."""

from __future__ import annotations

from typing import List, Optional

from .models import GranularityLevel, KnowledgeComponent, LearningObjective, QualityReport

_COMPLETENESS_FIELDS = ("examples", "non_examples", "observable_evidence", "practice_tasks")


def _kc_completeness(kc: KnowledgeComponent) -> float:
    filled = sum(1 for f in _COMPLETENESS_FIELDS if getattr(kc, f))
    return filled / len(_COMPLETENESS_FIELDS)


def compute_quality_score(
    kcs: List[KnowledgeComponent],
    los: List[LearningObjective],
    iteration: int = 0,
    issues: Optional[List[str]] = None,
    recommendations: Optional[List[str]] = None,
) -> QualityReport:
    if not kcs or not los:
        return QualityReport(
            iteration=iteration,
            overall_score=0.0,
            coverage_score=0.0,
            granularity_score=0.0,
            distinctiveness_score=0.0,
            completeness_score=0.0,
            total_kcs=len(kcs),
            total_los=len(los),
            issues=issues or [],
            recommendations=recommendations or [],
        )

    covered_lo_ids = {lo_id for kc in kcs for lo_id in kc.parent_lo_ids}
    lo_ids = {lo.id for lo in los}
    coverage = len(lo_ids & covered_lo_ids) / len(lo_ids)

    fine_or_atomic = sum(
        1 for kc in kcs if kc.granularity_level in (GranularityLevel.fine, GranularityLevel.atomic)
    )
    granularity = fine_or_atomic / len(kcs)

    with_matches = sum(
        1 for kc in kcs if any(r.type == "matches" for r in kc.relationships)
    )
    distinctiveness = 1.0 - (with_matches / len(kcs))

    completeness = sum(_kc_completeness(kc) for kc in kcs) / len(kcs)

    overall = (
        0.30 * coverage
        + 0.20 * granularity
        + 0.30 * distinctiveness
        + 0.20 * completeness
    )

    return QualityReport(
        iteration=iteration,
        overall_score=round(overall, 4),
        coverage_score=round(coverage, 4),
        granularity_score=round(granularity, 4),
        distinctiveness_score=round(distinctiveness, 4),
        completeness_score=round(completeness, 4),
        total_kcs=len(kcs),
        total_los=len(los),
        issues=issues or [],
        recommendations=recommendations or [],
    )
