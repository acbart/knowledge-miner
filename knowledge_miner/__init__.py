"""
Knowledge Component (KC) Decomposition Tool for Intelligent Tutoring Systems.

Public API – re-exports everything needed for backward compatibility.
"""

from .cli import build_arg_parser, create_llm_client
from .defaults import get_default_course_input
from .deduplication import EmbeddingSimilarity, compute_title_similarity, find_duplicate_kcs
from .id_utils import make_kc_id, make_lo_id, normalize_lo_ids
from .json_utils import (
    JSONParseError,
    _repair_pass1,
    _repair_pass2,
    _repair_pass3,
    _strip_code_fences,
    append_jsonl,
    load_json,
    load_jsonl,
    parse_json_with_repair,
    save_json,
)
from .kc_ops import (
    _apply_relationships,
    apply_merges,
    apply_refinement,
    fix_invalid_refs,
    select_subset,
    validate_relationships,
)
from .llm_clients import AnthropicClient, LLMClient, MockLLMClient, OpenAIClient
from .miner import KnowledgeMiner
from .models import (
    CourseInput,
    GranularityLevel,
    IterationLog,
    KCMap,
    KCType,
    KnowledgeComponent,
    LearningObjective,
    QualityReport,
    Relationship,
)
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

__all__ = [
    # models
    "CourseInput",
    "GranularityLevel",
    "IterationLog",
    "KCMap",
    "KCType",
    "KnowledgeComponent",
    "LearningObjective",
    "QualityReport",
    "Relationship",
    # id utils
    "make_kc_id",
    "make_lo_id",
    "normalize_lo_ids",
    # json utils
    "JSONParseError",
    "_repair_pass1",
    "_repair_pass2",
    "_repair_pass3",
    "_strip_code_fences",
    "append_jsonl",
    "load_json",
    "load_jsonl",
    "parse_json_with_repair",
    "save_json",
    # deduplication
    "EmbeddingSimilarity",
    "compute_title_similarity",
    "find_duplicate_kcs",
    # llm clients
    "AnthropicClient",
    "LLMClient",
    "MockLLMClient",
    "OpenAIClient",
    # prompts
    "CRITIQUE_SYSTEM_PROMPT",
    "DEDUPLICATION_SYSTEM_PROMPT",
    "GENERATION_SYSTEM_PROMPT",
    "REFINEMENT_SYSTEM_PROMPT",
    "RELATIONSHIP_SYSTEM_PROMPT",
    "build_critique_prompt",
    "build_deduplication_prompt",
    "build_generation_prompt",
    "build_refinement_prompt",
    "build_relationship_prompt",
    # kc ops
    "_apply_relationships",
    "apply_merges",
    "apply_refinement",
    "fix_invalid_refs",
    "select_subset",
    "validate_relationships",
    # quality
    "compute_quality_score",
    # state
    "RunState",
    # defaults
    "get_default_course_input",
    # miner
    "KnowledgeMiner",
    # cli
    "build_arg_parser",
    "create_llm_client",
]
