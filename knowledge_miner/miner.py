"""KnowledgeMiner: the main iterative KC decomposition runner."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .deduplication import EmbeddingSimilarity, find_duplicate_kcs
from .id_utils import make_kc_id
from .kc_ops import (
    _apply_relationships,
    apply_merges,
    apply_refinement,
    fix_invalid_refs,
    select_subset,
    validate_relationships,
)
from .llm_clients import LLMClient
from .models import CourseInput, IterationLog, KCMap, KnowledgeComponent, LearningObjective
from .json_utils import parse_json_with_repair
from .prompts import (
    CRITIQUE_SYSTEM_PROMPT,
    DEDUPLICATION_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    RELATIONSHIP_SYSTEM_PROMPT,
    build_critique_prompt,
    build_deduplication_prompt,
    build_generation_prompt,
    build_refinement_prompt,
    build_relationship_prompt,
)
from .quality import compute_quality_score
from .state import RunState

logger = logging.getLogger(__name__)


class KnowledgeMiner:
    def __init__(
        self,
        llm: LLMClient,
        state: RunState,
        max_iterations: int = 10,
        batch_size: int = 5,
        quality_threshold: float = 0.85,
        use_embeddings: bool = True,
    ) -> None:
        self.llm = llm
        self.state = state
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.quality_threshold = quality_threshold
        self._embedding_sim = EmbeddingSimilarity() if use_embeddings else None

    # --- LLM call wrappers ---

    def _generate(
        self,
        course_desc: str,
        objectives: List[LearningObjective],
        existing_kc_ids: List[str],
    ) -> List[dict]:
        response = self.llm.complete(
            GENERATION_SYSTEM_PROMPT,
            build_generation_prompt(course_desc, objectives, existing_kc_ids),
        )
        data = parse_json_with_repair(response)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("kcs", [])
        logger.warning("Unexpected generation response shape; returning []")
        return []

    def _critique(
        self,
        course_desc: str,
        objectives: List[LearningObjective],
        kcs: List[KnowledgeComponent],
    ) -> dict:
        response = self.llm.complete(
            CRITIQUE_SYSTEM_PROMPT,
            build_critique_prompt(course_desc, objectives, kcs),
        )
        return parse_json_with_repair(response)

    def _refine(
        self,
        course_desc: str,
        objectives: List[LearningObjective],
        kcs: List[KnowledgeComponent],
        critique: dict,
    ) -> dict:
        response = self.llm.complete(
            REFINEMENT_SYSTEM_PROMPT,
            build_refinement_prompt(course_desc, objectives, kcs, critique),
        )
        return parse_json_with_repair(response)

    def _extract_relationships(self, kcs: List[KnowledgeComponent]) -> dict:
        response = self.llm.complete(
            RELATIONSHIP_SYSTEM_PROMPT, build_relationship_prompt(kcs)
        )
        return parse_json_with_repair(response)

    def _deduplicate(
        self, kcs: List[KnowledgeComponent], pairs: List[Tuple[str, str]]
    ) -> dict:
        if not pairs:
            return {"merges": []}
        response = self.llm.complete(
            DEDUPLICATION_SYSTEM_PROMPT, build_deduplication_prompt(kcs, pairs)
        )
        return parse_json_with_repair(response)

    # --- Initial generation ---

    def _initial_generation(self, course_input: CourseInput) -> KCMap:
        logger.info(
            "Running initial KC generation for %d LOs...",
            len(course_input.learning_objectives),
        )
        raw_kcs = self._generate(
            course_input.course_description, course_input.learning_objectives, []
        )

        kcs_by_id: Dict[str, KnowledgeComponent] = {}
        existing_id_set: set = set()
        for kc_data in raw_kcs:
            if not isinstance(kc_data, dict):
                continue
            kc_id = kc_data.get("id") or make_kc_id(
                kc_data.get("title", "unnamed"), existing_id_set
            )
            kc_data["id"] = kc_id
            if kc_id not in kcs_by_id:
                try:
                    kcs_by_id[kc_id] = KnowledgeComponent.model_validate(kc_data)
                    existing_id_set.add(kc_id)
                except Exception as exc:
                    logger.warning("Skipping invalid KC %s: %s", kc_id, exc)

        logger.info("Initial generation produced %d KCs", len(kcs_by_id))
        return KCMap(
            course_description=course_input.course_description,
            kcs=list(kcs_by_id.values()),
        )

    # --- Main loop ---

    def run(self, course_input: CourseInput, start_iteration: int = 0) -> KCMap:
        los = course_input.learning_objectives

        kc_map = self.state.load_kc_map()
        if kc_map is None:
            kc_map = KCMap(course_description=course_input.course_description)

        if not kc_map.kcs:
            kc_map = self._initial_generation(course_input)
            kc_map.kcs = fix_invalid_refs(kc_map.kcs)
            self.state.save_kc_map(kc_map)

        # Save an initial quality snapshot before the iterative loop begins
        initial_quality = compute_quality_score(kc_map.kcs, los, start_iteration)
        self.state.save_quality_report(initial_quality)
        logger.info(
            "Initial quality: %.3f  (coverage=%.2f, gran=%.2f, distinct=%.2f, complete=%.2f)",
            initial_quality.overall_score,
            initial_quality.coverage_score,
            initial_quality.granularity_score,
            initial_quality.distinctiveness_score,
            initial_quality.completeness_score,
        )

        consecutive_no_improvement = 0

        for iteration in range(start_iteration, self.max_iterations):
            logger.info(
                "--- Iteration %d / %d  (%d KCs so far) ---",
                iteration + 1,
                self.max_iterations,
                len(kc_map.kcs),
            )

            quality_before = compute_quality_score(kc_map.kcs, los, iteration)

            subset_los, subset_kcs = select_subset(
                los, kc_map.kcs, self.batch_size, iteration
            )
            logger.debug("Subset: %d LOs, %d KCs", len(subset_los), len(subset_kcs))

            # Critique
            try:
                critique = self._critique(
                    course_input.course_description, subset_los, kc_map.kcs
                )
            except Exception as exc:
                logger.error("Critique call failed: %s", exc)
                critique = {
                    "issues": [],
                    "recommendations": [],
                    "coverage_score": 0.5,
                    "granularity_score": 0.5,
                    "distinctiveness_score": 0.5,
                }

            # Threshold check using critique-reported scores (per spec step 4)
            critique_score = (
                critique.get("coverage_score", 0.5)
                + critique.get("granularity_score", 0.5)
                + critique.get("distinctiveness_score", 0.5)
            ) / 3.0
            logger.info(
                "Critique score: %.3f (threshold: %.2f)",
                critique_score,
                self.quality_threshold,
            )

            if critique_score >= self.quality_threshold:
                logger.info("Quality threshold reached via critique. Stopping.")
                self.state.save_quality_report(
                    compute_quality_score(
                        kc_map.kcs,
                        los,
                        iteration,
                        issues=critique.get("issues", []),
                        recommendations=critique.get("recommendations", []),
                    )
                )
                break

            # Refinement
            try:
                refinement = self._refine(
                    course_input.course_description, subset_los, kc_map.kcs, critique
                )
            except Exception as exc:
                logger.error("Refinement call failed: %s", exc)
                refinement = {
                    "added": [],
                    "modified": [],
                    "removed": [],
                    "rationale": "failed",
                }

            existing_ids = {kc.id for kc in kc_map.kcs}
            kc_map, kcs_added, kcs_modified, kcs_removed = apply_refinement(
                kc_map, refinement, existing_ids
            )
            logger.info(
                "Changes: +%d added, ~%d modified, -%d removed",
                kcs_added,
                kcs_modified,
                kcs_removed,
            )

            # Validate and repair dangling references
            ref_errors = validate_relationships(kc_map.kcs)
            if ref_errors:
                logger.warning("%d relationship errors — repairing", len(ref_errors))
                for err in ref_errors[:5]:
                    logger.debug("  %s", err)
                kc_map.kcs = fix_invalid_refs(kc_map.kcs)

            # Relationship extraction when new KCs were added
            if kcs_added > 0:
                try:
                    rel_data = self._extract_relationships(kc_map.kcs)
                    _apply_relationships(kc_map, rel_data.get("relationships", []))
                except Exception as exc:
                    logger.warning("Relationship extraction failed: %s", exc)

            # Deduplication
            dup_pairs = find_duplicate_kcs(kc_map.kcs)
            if dup_pairs:
                logger.info("Found %d potential duplicate pair(s)", len(dup_pairs))
                try:
                    dedup = self._deduplicate(kc_map.kcs, dup_pairs)
                    merges = dedup.get("merges", [])
                    if merges:
                        kc_map = apply_merges(kc_map, merges)
                        logger.info("Merged %d pair(s)", len(merges))
                except Exception as exc:
                    logger.warning("Deduplication failed: %s", exc)

            # Quality after
            quality_after = compute_quality_score(
                kc_map.kcs,
                los,
                iteration,
                issues=critique.get("issues", []),
                recommendations=critique.get("recommendations", []),
            )
            logger.info(
                "Quality after:  %.3f  (coverage=%.2f, gran=%.2f, distinct=%.2f, complete=%.2f)",
                quality_after.overall_score,
                quality_after.coverage_score,
                quality_after.granularity_score,
                quality_after.distinctiveness_score,
                quality_after.completeness_score,
            )

            # Checkpoint
            self.state.save_kc_map(kc_map)
            self.state.save_quality_report(quality_after)

            log = IterationLog(
                iteration=iteration,
                timestamp=datetime.now(timezone.utc).isoformat(),
                subset_lo_ids=[lo.id for lo in subset_los],
                subset_kc_ids=[kc.id for kc in subset_kcs],
                kcs_added=kcs_added,
                kcs_modified=kcs_modified,
                kcs_removed=kcs_removed,
                improvements=critique.get("recommendations", [])[:5],
                rationale=refinement.get("rationale", ""),
                quality_score_before=quality_before.overall_score,
                quality_score_after=quality_after.overall_score,
            )
            self.state.append_iteration_log(log)

            if quality_after.overall_score >= self.quality_threshold:
                logger.info("Quality threshold reached after refinement. Stopping.")
                break

            if kcs_added + kcs_modified == 0:
                consecutive_no_improvement += 1
                logger.info(
                    "No KCs added/modified (%d consecutive)", consecutive_no_improvement
                )
                if consecutive_no_improvement >= 2:
                    logger.info("Stopping: 2 consecutive iterations without improvement.")
                    break
            else:
                consecutive_no_improvement = 0

        logger.info("Run complete. Final KC count: %d", len(kc_map.kcs))
        return kc_map
