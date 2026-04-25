"""KC map operations: validation, reference repair, refinement, merges, and subset selection."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .id_utils import make_kc_id
from .models import KCMap, KnowledgeComponent, LearningObjective, Relationship

logger = logging.getLogger(__name__)


def validate_relationships(kcs: List[KnowledgeComponent]) -> List[str]:
    kc_ids = {kc.id for kc in kcs}
    errors: List[str] = []
    for kc in kcs:
        for rel in kc.relationships:
            if rel.kc_id == kc.id:
                errors.append(f"{kc.id}: self-referential relationship")
            elif rel.kc_id not in kc_ids:
                errors.append(f"{kc.id}: relationship target '{rel.kc_id}' not found")
        for prereq in kc.prerequisites:
            if prereq == kc.id:
                errors.append(f"{kc.id}: self-referential prerequisite")
            elif prereq not in kc_ids:
                errors.append(f"{kc.id}: prerequisite '{prereq}' not found")
    return errors


def fix_invalid_refs(kcs: List[KnowledgeComponent]) -> List[KnowledgeComponent]:
    """Remove dangling and self-referential relationship/prerequisite entries."""
    kc_ids = {kc.id for kc in kcs}
    fixed: List[KnowledgeComponent] = []
    for kc in kcs:
        data = kc.model_dump()
        data["prerequisites"] = [
            p for p in data["prerequisites"] if p in kc_ids and p != kc.id
        ]
        data["relationships"] = [
            r for r in data["relationships"]
            if r["kc_id"] in kc_ids and r["kc_id"] != kc.id
        ]
        fixed.append(KnowledgeComponent.model_validate(data))
    return fixed


def apply_refinement(
    kc_map: KCMap,
    refinement: dict,
    existing_ids: Optional[set] = None,
) -> Tuple[KCMap, int, int, int]:
    """Apply added/modified/removed changes.  Returns (new_map, added, modified, removed)."""
    if existing_ids is None:
        existing_ids = {kc.id for kc in kc_map.kcs}

    kcs_by_id: Dict[str, KnowledgeComponent] = {kc.id: kc for kc in kc_map.kcs}

    removed_ids = set(refinement.get("removed", []))
    for rid in removed_ids:
        kcs_by_id.pop(rid, None)

    modified = 0
    for kc_data in refinement.get("modified", []):
        if not isinstance(kc_data, dict):
            continue
        kc_id = kc_data.get("id")
        if kc_id and kc_id in kcs_by_id:
            try:
                kcs_by_id[kc_id] = KnowledgeComponent.model_validate(kc_data)
                modified += 1
            except Exception as exc:
                logger.warning("Skipping invalid modified KC %s: %s", kc_id, exc)

    added = 0
    for kc_data in refinement.get("added", []):
        if not isinstance(kc_data, dict):
            continue
        kc_id = kc_data.get("id")
        if not kc_id:
            kc_id = make_kc_id(kc_data.get("title", "unnamed"), existing_ids)
            kc_data["id"] = kc_id
        if kc_id not in kcs_by_id:
            try:
                kcs_by_id[kc_id] = KnowledgeComponent.model_validate(kc_data)
                existing_ids.add(kc_id)
                added += 1
            except Exception as exc:
                logger.warning("Skipping invalid added KC %s: %s", kc_id, exc)
        else:
            logger.debug("KC %s already exists — skipping add", kc_id)

    return (
        KCMap(course_description=kc_map.course_description, kcs=list(kcs_by_id.values())),
        added,
        modified,
        len(removed_ids),
    )


def apply_merges(kc_map: KCMap, merges: List[dict]) -> KCMap:
    """Discard merged KCs and remap all references to their keep counterpart."""
    discard_to_keep: Dict[str, str] = {
        m["discard_id"]: m["keep_id"]
        for m in merges
        if "discard_id" in m and "keep_id" in m
    }
    discard_ids = set(discard_to_keep)

    new_kcs: List[KnowledgeComponent] = []
    for kc in kc_map.kcs:
        if kc.id in discard_ids:
            continue
        data = kc.model_dump()
        data["prerequisites"] = [discard_to_keep.get(p, p) for p in data["prerequisites"]]
        data["relationships"] = [
            {**r, "kc_id": discard_to_keep.get(r["kc_id"], r["kc_id"])}
            for r in data["relationships"]
            if r["kc_id"] not in discard_ids or r["kc_id"] in discard_to_keep
        ]
        new_kcs.append(KnowledgeComponent.model_validate(data))

    return KCMap(course_description=kc_map.course_description, kcs=new_kcs)


def _apply_relationships(kc_map: KCMap, relationships: List[dict]) -> None:
    """Merge externally extracted relationships into the KC map (in-place)."""
    kcs_by_id = {kc.id: kc for kc in kc_map.kcs}
    kc_ids = set(kcs_by_id)
    for rel in relationships:
        from_id = rel.get("from_kc_id", "")
        to_id = rel.get("kc_id", "")
        rel_type = rel.get("type", "")
        if from_id not in kc_ids or to_id not in kc_ids or from_id == to_id:
            continue
        if rel_type not in ("assumes", "extends", "matches"):
            continue
        kc = kcs_by_id[from_id]
        existing = {(r.type, r.kc_id) for r in kc.relationships}
        if (rel_type, to_id) not in existing:
            kc.relationships.append(Relationship(type=rel_type, kc_id=to_id))


def select_subset(
    los: List[LearningObjective],
    kcs: List[KnowledgeComponent],
    batch_size: int,
    iteration: int,
) -> Tuple[List[LearningObjective], List[KnowledgeComponent]]:
    """Rotate through LOs in batches; include KCs that belong to the selected LOs."""
    n_los = max(len(los), 1)
    lo_start = (iteration * batch_size) % n_los
    subset_los = los[lo_start : lo_start + batch_size] or los[:batch_size]

    subset_lo_ids = {lo.id for lo in subset_los}
    subset_kcs = [kc for kc in kcs if set(kc.parent_lo_ids) & subset_lo_ids]

    # Top up with unassigned KCs so the LLM sees sufficient context
    remaining_budget = batch_size * 3 - len(subset_kcs)
    if remaining_budget > 0:
        extras = [kc for kc in kcs if kc not in subset_kcs][:remaining_budget]
        subset_kcs.extend(extras)

    return subset_los, subset_kcs
