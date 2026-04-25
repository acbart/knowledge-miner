"""Enums and Pydantic data models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field


class KCType(str, Enum):
    concept = "concept"
    procedure = "procedure"
    skill = "skill"
    strategy = "strategy"
    misconception = "misconception"
    debugging_skill = "debugging_skill"
    notation = "notation"
    tool_practical = "tool_practical"


class GranularityLevel(str, Enum):
    atomic = "atomic"
    fine = "fine"
    medium = "medium"
    coarse = "coarse"


class Relationship(BaseModel):
    type: Literal["assumes", "extends", "matches"]
    kc_id: str


class LearningObjective(BaseModel):
    id: str
    title: str
    description: str = ""


class KnowledgeComponent(BaseModel):
    id: str
    title: str
    description: str
    parent_lo_ids: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    type: KCType
    granularity_level: GranularityLevel
    examples: List[str] = Field(default_factory=list)
    non_examples: List[str] = Field(default_factory=list)
    common_misconceptions: List[str] = Field(default_factory=list)
    observable_evidence: List[str] = Field(default_factory=list)
    likely_errors: List[str] = Field(default_factory=list)
    practice_tasks: List[str] = Field(default_factory=list)
    assessment_cues: List[str] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


class IterationLog(BaseModel):
    iteration: int
    timestamp: str
    subset_lo_ids: List[str]
    subset_kc_ids: List[str]
    kcs_added: int
    kcs_modified: int
    kcs_removed: int
    improvements: List[str]
    rationale: str
    quality_score_before: float
    quality_score_after: float


class QualityReport(BaseModel):
    iteration: int
    overall_score: float
    coverage_score: float
    granularity_score: float
    distinctiveness_score: float
    completeness_score: float
    total_kcs: int
    total_los: int
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CourseInput(BaseModel):
    course_description: str
    learning_objectives: List[LearningObjective]


class KCMap(BaseModel):
    course_description: str
    kcs: List[KnowledgeComponent] = Field(default_factory=list)
    last_updated: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
