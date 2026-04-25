"""Command-line interface for the Knowledge Miner tool."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .defaults import get_default_course_input
from .id_utils import normalize_lo_ids
from .json_utils import load_json
from .llm_clients import AnthropicClient, LLMClient, MockLLMClient, OpenAIClient
from .miner import KnowledgeMiner
from .models import CourseInput, KCMap
from .state import RunState

logger = logging.getLogger(__name__)


def create_llm_client(
    provider: str, model: str | None, api_key: str | None
) -> LLMClient:
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenAI API key required (--api-key or OPENAI_API_KEY env var)"
            )
        return OpenAIClient(api_key=key, model=model or "gpt-4o-mini")
    if provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "Anthropic API key required (--api-key or ANTHROPIC_API_KEY env var)"
            )
        return AnthropicClient(api_key=key, model=model or "claude-3-haiku-20240307")
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Choose openai, anthropic, or mock."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Knowledge Component Decomposition Tool for ITS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--course-desc", "-d", default=None, help="Course description string")
    p.add_argument(
        "--input-file", "-i", default=None,
        help="JSON file with course_description and learning_objectives",
    )
    p.add_argument("--output-dir", "-o", default="./output", help="Output directory")
    p.add_argument("--max-iterations", "-n", type=int, default=10, help="Max iterations")
    p.add_argument(
        "--batch-size", "-b", type=int, default=5,
        help="Number of LOs/KCs per LLM call",
    )
    p.add_argument(
        "--llm-provider", default="mock",
        choices=["openai", "anthropic", "mock"],
        help="LLM provider",
    )
    p.add_argument("--model", default=None, help="Model name override")
    p.add_argument("--api-key", default=None, help="API key")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Use mock LLM (same as --llm-provider mock)",
    )
    p.add_argument(
        "--quality-threshold", type=float, default=0.85,
        help="Stop when quality score exceeds this value",
    )
    p.add_argument("--no-embeddings", action="store_true", help="Disable embeddings similarity")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p.add_argument("--resume", action="store_true", help="Resume from existing output directory")
    return p


def _load_course_input(args: argparse.Namespace) -> CourseInput:
    if args.input_file:
        raw = load_json(Path(args.input_file))
        ci = CourseInput.model_validate(raw)
        ci.learning_objectives = normalize_lo_ids(ci.learning_objectives)
        return ci
    if args.course_desc:
        return CourseInput(course_description=args.course_desc, learning_objectives=[])
    return get_default_course_input()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        args.llm_provider = "mock"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(output_dir)

    start_iteration = 0
    if args.resume and state.course_input_path.exists():
        logger.info("Resuming from %s", output_dir)
        course_input = state.load_course_input()
        logs = state.load_iteration_logs()
        if logs:
            start_iteration = max(log.iteration for log in logs) + 1
        logger.info("Resuming at iteration %d", start_iteration)
    else:
        course_input = _load_course_input(args)
        state.save_course_input(course_input)

    logger.info("Course: %s", course_input.course_description)
    logger.info("Learning objectives: %d", len(course_input.learning_objectives))

    try:
        llm = create_llm_client(args.llm_provider, args.model, args.api_key)
    except (ValueError, ImportError) as exc:
        logger.error("Cannot create LLM client: %s", exc)
        sys.exit(1)

    miner = KnowledgeMiner(
        llm=llm,
        state=state,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        quality_threshold=args.quality_threshold,
        use_embeddings=not args.no_embeddings,
    )

    kc_map: KCMap
    try:
        kc_map = miner.run(course_input, start_iteration=start_iteration)
    except KeyboardInterrupt:
        logger.info("Interrupted — saving checkpoint...")
        saved = state.load_kc_map()
        if saved:
            logger.info("Checkpoint preserved with %d KCs.", len(saved.kcs))
        sys.exit(0)

    quality_report = state.load_quality_report()

    print("\n=== Final Quality Report ===")
    if quality_report:
        print(f"  Overall score:      {quality_report.overall_score:.3f}")
        print(f"  Coverage score:     {quality_report.coverage_score:.3f}")
        print(f"  Granularity score:  {quality_report.granularity_score:.3f}")
        print(f"  Distinctiveness:    {quality_report.distinctiveness_score:.3f}")
        print(f"  Completeness:       {quality_report.completeness_score:.3f}")
        print(f"  Total KCs:          {quality_report.total_kcs}")
        print(f"  Total LOs:          {quality_report.total_los}")
        if quality_report.issues:
            print("\n  Issues:")
            for issue in quality_report.issues[:5]:
                print(f"    - {issue}")
        if quality_report.recommendations:
            print("\n  Recommendations:")
            for rec in quality_report.recommendations[:5]:
                print(f"    - {rec}")
    else:
        print("  (no quality report generated)")

    print(f"\nOutput directory: {output_dir.resolve()}")
    print(f"  kc_map.json            — {len(kc_map.kcs)} knowledge components")
    print(f"  course_input.json      — course description + LOs")
    print(f"  iteration_logs.jsonl   — per-iteration log entries")
    print(f"  quality_report.json    — final quality metrics")
