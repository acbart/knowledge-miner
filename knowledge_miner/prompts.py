"""LLM prompt templates and builder functions."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from .models import KnowledgeComponent, LearningObjective


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_PROMPT = """\
You are an expert instructional designer specializing in Knowledge Component (KC) \
decomposition for intelligent tutoring systems (ITS).

Your task: decompose course learning objectives into exhaustive, well-structured KCs.

Guidelines:
- Avoid vague KCs such as "understand loops". Prefer teachable, assessable, observable ones.
- Distinguish concept / procedure / misconception / debugging_skill / tool_practical / skill / strategy / notation.
- Include programming-specific issues: syntax errors, tracing, debugging, IDE behavior, error
  messages, test cases, I/O, mental models of execution.
- Only list DIRECTLY required prerequisites (avoid infinite expansion).
- Each KC must have non-empty: examples, non_examples, observable_evidence, practice_tasks.
- IDs must be deterministic slugs like "kc-variable-assignment".

Return a JSON **array** of KC objects matching this exact schema:
{
  "id": "kc-<slug>",
  "title": "...",
  "description": "...",
  "parent_lo_ids": ["lo-001"],
  "prerequisites": ["kc-..."],
  "type": "concept|procedure|skill|strategy|misconception|debugging_skill|notation|tool_practical",
  "granularity_level": "atomic|fine|medium|coarse",
  "examples": [...],
  "non_examples": [...],
  "common_misconceptions": [...],
  "observable_evidence": [...],
  "likely_errors": [...],
  "practice_tasks": [...],
  "assessment_cues": [...],
  "relationships": [{"type": "assumes|extends|matches", "kc_id": "kc-..."}]
}

Return ONLY valid JSON. No explanation text.\
"""

CRITIQUE_SYSTEM_PROMPT = """\
You are an expert in Knowledge Component analysis for intelligent tutoring systems.

Evaluate the provided KC set against the learning objectives on these dimensions:
1. Coverage — Are all LOs represented by at least one KC?
2. Validity — Are KCs specific, observable, and assessable?
3. Distinctiveness — Are KCs sufficiently distinct with no redundancy?
4. Granularity — Are KCs at fine/atomic level (preferred)?
5. Prerequisite boundedness — Are prerequisites reasonable and non-circular?
6. ITS usefulness — Do KCs include misconceptions, debugging skills, practical tasks?

Return JSON with EXACTLY this structure (no extra keys):
{
  "issues": ["issue 1", ...],
  "recommendations": ["rec 1", ...],
  "coverage_score": <float 0-1>,
  "granularity_score": <float 0-1>,
  "distinctiveness_score": <float 0-1>
}

Return ONLY valid JSON.\
"""

REFINEMENT_SYSTEM_PROMPT = """\
You are an expert instructional designer refining a Knowledge Component map for an ITS.

Given KCs and a critique, propose incremental improvements:
- Add missing KCs (full KC objects)
- Modify existing KCs (full updated KC objects)
- Remove redundant KCs (by ID string)
- Merge near-duplicates using "matches" relationships

Return JSON with EXACTLY this structure:
{
  "added": [<full KC objects>],
  "modified": [<full KC objects>],
  "removed": ["kc-id-1", ...],
  "rationale": "Brief explanation"
}

Return ONLY valid JSON.\
"""

RELATIONSHIP_SYSTEM_PROMPT = """\
You are an expert in prerequisite analysis for intelligent tutoring systems.

Given Knowledge Components, identify relationships:
- assumes: KC A requires KC B (B must be learned before A)
- extends: KC A builds on KC B
- matches: KC A and B cover the same concept (candidate for merge)

Return JSON:
{
  "relationships": [
    {"from_kc_id": "kc-...", "type": "assumes|extends|matches", "kc_id": "kc-..."},
    ...
  ]
}

Return ONLY valid JSON.\
"""

DEDUPLICATION_SYSTEM_PROMPT = """\
You are an expert in Knowledge Component analysis.

Given pairs of potentially duplicate KCs, decide whether to merge or keep them.
Merge when they cover exactly the same concept or one is a strict subset of the other.

Return JSON:
{
  "merges": [
    {"keep_id": "kc-...", "discard_id": "kc-...", "reason": "..."},
    ...
  ]
}

Return ONLY valid JSON.\
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_generation_prompt(
    course_desc: str,
    objectives: List[LearningObjective],
    existing_kc_ids: List[str],
) -> str:
    lo_lines = "\n".join(
        f"  - {lo.id}: {lo.title}" + (f" — {lo.description}" if lo.description else "")
        for lo in objectives
    )
    existing = ", ".join(existing_kc_ids) if existing_kc_ids else "none"
    return (
        f"Course: {course_desc}\n\n"
        f"Learning Objectives to decompose:\n{lo_lines}\n\n"
        f"Already-defined KC IDs (do not duplicate): {existing}\n\n"
        "Generate an exhaustive list of Knowledge Components. Return a JSON array."
    )


def build_critique_prompt(
    course_desc: str,
    objectives: List[LearningObjective],
    kcs: List[KnowledgeComponent],
) -> str:
    lo_lines = "\n".join(f"  - {lo.id}: {lo.title}" for lo in objectives)
    kc_summary = json.dumps(
        [
            {
                "id": kc.id,
                "title": kc.title,
                "type": kc.type,
                "granularity_level": kc.granularity_level,
                "parent_lo_ids": kc.parent_lo_ids,
                "prerequisites": kc.prerequisites,
                "has_examples": bool(kc.examples),
                "has_observable_evidence": bool(kc.observable_evidence),
            }
            for kc in kcs
        ],
        indent=2,
    )
    return (
        f"Course: {course_desc}\n\n"
        f"Learning Objectives:\n{lo_lines}\n\n"
        f"Current Knowledge Components:\n{kc_summary}\n\n"
        "Critique this KC set and return JSON."
    )


def build_refinement_prompt(
    course_desc: str,
    objectives: List[LearningObjective],
    kcs: List[KnowledgeComponent],
    critique: dict,
) -> str:
    lo_lines = "\n".join(f"  - {lo.id}: {lo.title}" for lo in objectives)
    kc_text = json.dumps([kc.model_dump() for kc in kcs], indent=2)
    critique_text = json.dumps(critique, indent=2)
    return (
        f"Course: {course_desc}\n\n"
        f"Learning Objectives:\n{lo_lines}\n\n"
        f"Current KCs:\n{kc_text}\n\n"
        f"Critique:\n{critique_text}\n\n"
        "Propose improvements. Return JSON with added/modified/removed/rationale."
    )


def build_relationship_prompt(kcs: List[KnowledgeComponent]) -> str:
    summary = json.dumps(
        [{"id": kc.id, "title": kc.title, "description": kc.description[:120]} for kc in kcs],
        indent=2,
    )
    return f"Knowledge Components:\n{summary}\n\nIdentify relationships. Return JSON."


def build_deduplication_prompt(
    kcs: List[KnowledgeComponent], pairs: List[Tuple[str, str]]
) -> str:
    kc_by_id = {kc.id: kc for kc in kcs}
    pair_data = [
        {"kc1": kc_by_id[a].model_dump(), "kc2": kc_by_id[b].model_dump()}
        for a, b in pairs
        if a in kc_by_id and b in kc_by_id
    ]
    return (
        f"Potentially duplicate KC pairs:\n{json.dumps(pair_data, indent=2)}\n\n"
        "Decide merges. Return JSON."
    )
