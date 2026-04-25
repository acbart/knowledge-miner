"""Unit tests for knowledge_miner.py."""

import json
import tempfile
from pathlib import Path

import pytest

from knowledge_miner import (
    GranularityLevel,
    JSONParseError,
    KCMap,
    KCType,
    KnowledgeComponent,
    KnowledgeMiner,
    LearningObjective,
    MockLLMClient,
    QualityReport,
    Relationship,
    RunState,
    _apply_relationships,
    _repair_pass1,
    _repair_pass2,
    _repair_pass3,
    _strip_code_fences,
    apply_merges,
    apply_refinement,
    build_critique_prompt,
    build_deduplication_prompt,
    build_generation_prompt,
    build_relationship_prompt,
    build_refinement_prompt,
    compute_quality_score,
    compute_title_similarity,
    find_duplicate_kcs,
    fix_invalid_refs,
    get_default_course_input,
    make_kc_id,
    make_lo_id,
    normalize_lo_ids,
    parse_json_with_repair,
    select_subset,
    validate_relationships,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_lo(idx: int, title: str = "") -> LearningObjective:
    return LearningObjective(id=make_lo_id(idx), title=title or f"LO {idx}")


def _make_kc(kc_id: str, lo_ids=None, prereqs=None, gran=GranularityLevel.fine) -> KnowledgeComponent:
    return KnowledgeComponent(
        id=kc_id,
        title=kc_id.replace("kc-", "").replace("-", " ").title(),
        description="A test KC.",
        parent_lo_ids=lo_ids or ["lo-001"],
        prerequisites=prereqs or [],
        type=KCType.concept,
        granularity_level=gran,
        examples=["example"],
        non_examples=["non-example"],
        common_misconceptions=[],
        observable_evidence=["evidence"],
        likely_errors=[],
        practice_tasks=["task"],
        assessment_cues=[],
        relationships=[],
    )


# ---------------------------------------------------------------------------
# ID Generation
# ---------------------------------------------------------------------------


class TestMakeLoId:
    def test_zero_padded(self):
        assert make_lo_id(1) == "lo-001"
        assert make_lo_id(10) == "lo-010"
        assert make_lo_id(100) == "lo-100"


class TestMakeKcId:
    def test_basic_slug(self):
        assert make_kc_id("Variable Assignment", set()) == "kc-variable-assignment"

    def test_special_chars_replaced(self):
        result = make_kc_id("if/elif/else", set())
        assert result.startswith("kc-")
        assert "/" not in result

    def test_collision_gets_suffix(self):
        existing = {"kc-loops"}
        result = make_kc_id("loops", existing)
        assert result == "kc-loops-2"

    def test_multiple_collisions(self):
        existing = {"kc-loops", "kc-loops-2"}
        result = make_kc_id("loops", existing)
        assert result == "kc-loops-3"


class TestNormalizeLoIds:
    def test_ids_assigned_sequentially(self):
        los = [LearningObjective(id="", title=f"LO {i}") for i in range(3)]
        normalized = normalize_lo_ids(los)
        assert [lo.id for lo in normalized] == ["lo-001", "lo-002", "lo-003"]

    def test_titles_preserved(self):
        los = [LearningObjective(id="", title="Custom Title")]
        assert normalize_lo_ids(los)[0].title == "Custom Title"


# ---------------------------------------------------------------------------
# JSON Parsing & Repair
# ---------------------------------------------------------------------------


class TestStripCodeFences:
    def test_strips_json_fence(self):
        assert _strip_code_fences("```json\n[1,2,3]\n```") == "[1,2,3]"

    def test_strips_plain_fence(self):
        assert _strip_code_fences("```\n[1]\n```") == "[1]"

    def test_no_fence_unchanged(self):
        assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


class TestRepairPasses:
    def test_pass1_trailing_comma_object(self):
        result = _repair_pass1('{"a": 1,}')
        assert json.loads(result) == {"a": 1}

    def test_pass1_trailing_comma_array(self):
        result = _repair_pass1("[1, 2, 3,]")
        assert json.loads(result) == [1, 2, 3]

    def test_pass2_single_quotes(self):
        result = _repair_pass2("{'key': 'value'}")
        assert json.loads(result) == {"key": "value"}

    def test_pass3_truncates_at_last_delimiter(self):
        truncated = _repair_pass3('{"a": 1} trailing garbage')
        assert json.loads(truncated) == {"a": 1}


class TestParseJsonWithRepair:
    def test_clean_json(self):
        assert parse_json_with_repair("[1, 2, 3]") == [1, 2, 3]

    def test_code_fence(self):
        assert parse_json_with_repair('```json\n{"x": 1}\n```') == {"x": 1}

    def test_trailing_comma(self):
        assert parse_json_with_repair('{"a": 1,}') == {"a": 1}

    def test_invalid_json_raises(self):
        with pytest.raises(JSONParseError):
            parse_json_with_repair("this is not json at all!!!")


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestTitleSimilarity:
    def test_identical(self):
        assert compute_title_similarity("Hello", "Hello") == 1.0

    def test_completely_different(self):
        assert compute_title_similarity("loops", "functions") < 0.5

    def test_similar(self):
        score = compute_title_similarity("Variable Assignment", "Variable Assignments")
        assert score > 0.90


class TestFindDuplicateKcs:
    def test_no_duplicates(self):
        kcs = [_make_kc("kc-loops"), _make_kc("kc-functions")]
        assert find_duplicate_kcs(kcs) == []

    def test_finds_near_duplicate(self):
        kcs = [
            _make_kc("kc-loops"),
            KnowledgeComponent(
                id="kc-loops-copy",
                title="Loops",  # identical title to kc-loops KC title
                description="same",
                parent_lo_ids=["lo-001"],
                prerequisites=[],
                type=KCType.concept,
                granularity_level=GranularityLevel.fine,
                examples=["x"],
                non_examples=["y"],
                common_misconceptions=[],
                observable_evidence=["z"],
                likely_errors=[],
                practice_tasks=["t"],
                assessment_cues=[],
                relationships=[],
            ),
        ]
        # Both have the same title string "Loops"
        pairs = find_duplicate_kcs(kcs)
        assert len(pairs) == 1

    def test_threshold_respected(self):
        kcs = [_make_kc("kc-loops"), _make_kc("kc-functions")]
        assert find_duplicate_kcs(kcs, threshold=0.2) != []


# ---------------------------------------------------------------------------
# Relationship Validation
# ---------------------------------------------------------------------------


class TestValidateRelationships:
    def test_valid_relationships(self):
        kc_a = _make_kc("kc-a")
        kc_b = _make_kc("kc-b", prereqs=["kc-a"])
        kc_b.relationships.append(Relationship(type="assumes", kc_id="kc-a"))
        errors = validate_relationships([kc_a, kc_b])
        assert errors == []

    def test_missing_prerequisite(self):
        kc = _make_kc("kc-a", prereqs=["kc-nonexistent"])
        errors = validate_relationships([kc])
        assert any("kc-nonexistent" in e for e in errors)

    def test_missing_relationship_target(self):
        kc = _make_kc("kc-a")
        kc.relationships.append(Relationship(type="extends", kc_id="kc-ghost"))
        errors = validate_relationships([kc])
        assert any("kc-ghost" in e for e in errors)

    def test_self_referential_relationship(self):
        kc = _make_kc("kc-a")
        kc.relationships.append(Relationship(type="assumes", kc_id="kc-a"))
        errors = validate_relationships([kc])
        assert any("self-referential" in e for e in errors)

    def test_self_referential_prerequisite(self):
        kc = _make_kc("kc-a", prereqs=["kc-a"])
        errors = validate_relationships([kc])
        assert any("self-referential" in e for e in errors)


class TestFixInvalidRefs:
    def test_removes_dangling_prerequisites(self):
        kc = _make_kc("kc-a", prereqs=["kc-nonexistent"])
        fixed = fix_invalid_refs([kc])
        assert fixed[0].prerequisites == []

    def test_removes_dangling_relationship_targets(self):
        kc = _make_kc("kc-a")
        kc.relationships.append(Relationship(type="assumes", kc_id="kc-ghost"))
        fixed = fix_invalid_refs([kc])
        assert fixed[0].relationships == []

    def test_keeps_valid_refs(self):
        kc_a = _make_kc("kc-a")
        kc_b = _make_kc("kc-b", prereqs=["kc-a"])
        kc_b.relationships.append(Relationship(type="extends", kc_id="kc-a"))
        fixed = fix_invalid_refs([kc_a, kc_b])
        assert fixed[1].prerequisites == ["kc-a"]
        assert fixed[1].relationships[0].kc_id == "kc-a"


# ---------------------------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------------------------


class TestComputeQualityScore:
    def test_empty_kcs(self):
        los = [_make_lo(1)]
        report = compute_quality_score([], los)
        assert report.overall_score == 0.0

    def test_full_coverage(self):
        lo = _make_lo(1)
        kc = _make_kc("kc-a", lo_ids=["lo-001"])
        report = compute_quality_score([kc], [lo])
        assert report.coverage_score == 1.0

    def test_partial_coverage(self):
        los = [_make_lo(1), _make_lo(2)]
        kc = _make_kc("kc-a", lo_ids=["lo-001"])
        report = compute_quality_score([kc], los)
        assert report.coverage_score == 0.5

    def test_granularity_score(self):
        los = [_make_lo(1)]
        kc_fine = _make_kc("kc-fine", gran=GranularityLevel.fine)
        kc_coarse = _make_kc("kc-coarse", gran=GranularityLevel.coarse)
        report = compute_quality_score([kc_fine, kc_coarse], los)
        assert report.granularity_score == 0.5

    def test_distinctiveness_score_with_matches(self):
        los = [_make_lo(1)]
        kc_a = _make_kc("kc-a")
        kc_b = _make_kc("kc-b")
        kc_a.relationships.append(Relationship(type="matches", kc_id="kc-b"))
        report = compute_quality_score([kc_a, kc_b], los)
        assert report.distinctiveness_score < 1.0

    def test_report_fields_populated(self):
        los = [_make_lo(1)]
        kc = _make_kc("kc-a", lo_ids=["lo-001"])
        report = compute_quality_score([kc], los, iteration=3, issues=["iss1"])
        assert report.iteration == 3
        assert "iss1" in report.issues
        assert report.total_kcs == 1
        assert report.total_los == 1


# ---------------------------------------------------------------------------
# KC Map Operations
# ---------------------------------------------------------------------------


class TestApplyRefinement:
    def _base_map(self) -> KCMap:
        return KCMap(
            course_description="Test Course",
            kcs=[_make_kc("kc-existing")],
        )

    def test_add_new_kc(self):
        kc_map = self._base_map()
        new_kc = _make_kc("kc-new").model_dump()
        result, added, modified, removed = apply_refinement(
            kc_map, {"added": [new_kc], "modified": [], "removed": []}
        )
        assert added == 1
        assert any(kc.id == "kc-new" for kc in result.kcs)

    def test_modify_existing_kc(self):
        kc_map = self._base_map()
        updated = _make_kc("kc-existing").model_dump()
        updated["description"] = "Updated description"
        result, added, modified, removed = apply_refinement(
            kc_map, {"added": [], "modified": [updated], "removed": []}
        )
        assert modified == 1
        assert result.kcs[0].description == "Updated description"

    def test_remove_kc(self):
        kc_map = self._base_map()
        result, added, modified, removed = apply_refinement(
            kc_map, {"added": [], "modified": [], "removed": ["kc-existing"]}
        )
        assert removed == 1
        assert not any(kc.id == "kc-existing" for kc in result.kcs)

    def test_duplicate_add_is_skipped(self):
        kc_map = self._base_map()
        existing_kc = _make_kc("kc-existing").model_dump()
        result, added, _, _ = apply_refinement(
            kc_map, {"added": [existing_kc], "modified": [], "removed": []}
        )
        assert added == 0

    def test_invalid_kc_data_is_skipped(self):
        kc_map = self._base_map()
        bad_kc = {"id": "kc-bad", "type": "invalid_type_xxx"}
        result, added, _, _ = apply_refinement(
            kc_map, {"added": [bad_kc], "modified": [], "removed": []}
        )
        assert added == 0


class TestApplyMerges:
    def test_discards_merged_kc(self):
        kc_a = _make_kc("kc-a")
        kc_b = _make_kc("kc-b")
        kc_map = KCMap(course_description="C", kcs=[kc_a, kc_b])
        result = apply_merges(kc_map, [{"keep_id": "kc-a", "discard_id": "kc-b", "reason": "dup"}])
        assert len(result.kcs) == 1
        assert result.kcs[0].id == "kc-a"

    def test_remaps_references_to_discarded_kc(self):
        kc_a = _make_kc("kc-a")
        kc_b = _make_kc("kc-b")
        kc_c = _make_kc("kc-c", prereqs=["kc-b"])
        kc_map = KCMap(course_description="C", kcs=[kc_a, kc_b, kc_c])
        result = apply_merges(kc_map, [{"keep_id": "kc-a", "discard_id": "kc-b", "reason": "dup"}])
        c = next(kc for kc in result.kcs if kc.id == "kc-c")
        assert "kc-a" in c.prerequisites
        assert "kc-b" not in c.prerequisites


class TestApplyRelationships:
    def test_adds_new_relationship(self):
        kc_a = _make_kc("kc-a")
        kc_b = _make_kc("kc-b")
        kc_map = KCMap(course_description="C", kcs=[kc_a, kc_b])
        _apply_relationships(kc_map, [{"from_kc_id": "kc-b", "type": "assumes", "kc_id": "kc-a"}])
        kc_b_updated = next(kc for kc in kc_map.kcs if kc.id == "kc-b")
        assert any(r.kc_id == "kc-a" and r.type == "assumes" for r in kc_b_updated.relationships)

    def test_ignores_missing_kc_ids(self):
        kc_a = _make_kc("kc-a")
        kc_map = KCMap(course_description="C", kcs=[kc_a])
        # Should not raise; just skips invalid refs
        _apply_relationships(kc_map, [{"from_kc_id": "kc-ghost", "type": "assumes", "kc_id": "kc-a"}])

    def test_does_not_duplicate_relationship(self):
        kc_a = _make_kc("kc-a")
        kc_b = _make_kc("kc-b")
        kc_b.relationships.append(Relationship(type="assumes", kc_id="kc-a"))
        kc_map = KCMap(course_description="C", kcs=[kc_a, kc_b])
        _apply_relationships(kc_map, [{"from_kc_id": "kc-b", "type": "assumes", "kc_id": "kc-a"}])
        kc_b_updated = next(kc for kc in kc_map.kcs if kc.id == "kc-b")
        assumes = [r for r in kc_b_updated.relationships if r.type == "assumes" and r.kc_id == "kc-a"]
        assert len(assumes) == 1


# ---------------------------------------------------------------------------
# Batch Selection
# ---------------------------------------------------------------------------


class TestSelectSubset:
    def test_returns_correct_subset_size(self):
        los = [_make_lo(i) for i in range(1, 11)]
        kcs = [_make_kc(f"kc-{i}", lo_ids=[f"lo-{i:03d}"]) for i in range(1, 11)]
        subset_los, subset_kcs = select_subset(los, kcs, batch_size=3, iteration=0)
        assert len(subset_los) == 3

    def test_rotates_through_los(self):
        los = [_make_lo(i) for i in range(1, 6)]
        kcs = []
        # iteration=0 starts at index 0
        los0, _ = select_subset(los, kcs, batch_size=2, iteration=0)
        # iteration=1 starts at index 2
        los1, _ = select_subset(los, kcs, batch_size=2, iteration=1)
        assert los0[0].id != los1[0].id


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------


class TestPromptBuilders:
    def _los(self):
        return [_make_lo(1, "Variables"), _make_lo(2, "Loops")]

    def _kcs(self):
        return [_make_kc("kc-variables", lo_ids=["lo-001"])]

    def test_generation_prompt_contains_course(self):
        prompt = build_generation_prompt("TestCourse", self._los(), [])
        assert "TestCourse" in prompt

    def test_generation_prompt_contains_lo_ids(self):
        prompt = build_generation_prompt("C", self._los(), [])
        assert "lo-001" in prompt

    def test_critique_prompt_contains_kc_ids(self):
        prompt = build_critique_prompt("C", self._los(), self._kcs())
        assert "kc-variables" in prompt

    def test_refinement_prompt_contains_critique(self):
        critique = {"issues": ["issue1"], "recommendations": []}
        prompt = build_refinement_prompt("C", self._los(), self._kcs(), critique)
        assert "issue1" in prompt

    def test_relationship_prompt_contains_kc_titles(self):
        prompt = build_relationship_prompt(self._kcs())
        assert "Variables" in prompt

    def test_deduplication_prompt_contains_kc_data(self):
        kcs = self._kcs() + [_make_kc("kc-vars-2", lo_ids=["lo-001"])]
        pairs = [("kc-variables", "kc-vars-2")]
        prompt = build_deduplication_prompt(kcs, pairs)
        assert "kc-variables" in prompt


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------


class TestMockLLMClient:
    def _mock(self):
        return MockLLMClient()

    def test_generation_returns_list(self):
        mock = self._mock()
        response = mock.complete("You are an expert ... Return a JSON **array**", "Generate KCs.")
        data = parse_json_with_repair(response)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_critique_returns_scores(self):
        mock = self._mock()
        response = mock.complete("Return JSON with ... coverage_score ...", "Critique these KCs.")
        data = parse_json_with_repair(response)
        assert "coverage_score" in data
        assert "granularity_score" in data

    def test_refinement_returns_structure(self):
        mock = self._mock()
        response = mock.complete('"added" ... "modified" ... "removed"', "Refine these KCs.")
        data = parse_json_with_repair(response)
        assert "added" in data
        assert "modified" in data
        assert "removed" in data
        assert "rationale" in data

    def test_relationship_returns_list(self):
        mock = self._mock()
        response = mock.complete('"from_kc_id"', "Extract relationships.")
        data = parse_json_with_repair(response)
        assert "relationships" in data

    def test_deduplication_returns_merges(self):
        mock = self._mock()
        response = mock.complete('"discard_id"', "Deduplicate.")
        data = parse_json_with_repair(response)
        assert "merges" in data


# ---------------------------------------------------------------------------
# Default Course Input
# ---------------------------------------------------------------------------


class TestGetDefaultCourseInput:
    def test_returns_10_objectives(self):
        ci = get_default_course_input()
        assert len(ci.learning_objectives) == 10

    def test_ids_are_assigned(self):
        ci = get_default_course_input()
        for lo in ci.learning_objectives:
            assert lo.id.startswith("lo-")

    def test_course_description_set(self):
        ci = get_default_course_input()
        assert "Python" in ci.course_description


# ---------------------------------------------------------------------------
# RunState (file I/O)
# ---------------------------------------------------------------------------


class TestRunState:
    def _state(self, tmp_path):
        return RunState(tmp_path)

    def test_save_and_load_course_input(self, tmp_path):
        state = self._state(tmp_path)
        ci = get_default_course_input()
        state.save_course_input(ci)
        loaded = state.load_course_input()
        assert loaded.course_description == ci.course_description
        assert len(loaded.learning_objectives) == len(ci.learning_objectives)

    def test_save_and_load_kc_map(self, tmp_path):
        state = self._state(tmp_path)
        kc_map = KCMap(course_description="Test", kcs=[_make_kc("kc-test")])
        state.save_kc_map(kc_map)
        loaded = state.load_kc_map()
        assert loaded is not None
        assert len(loaded.kcs) == 1
        assert loaded.kcs[0].id == "kc-test"

    def test_load_kc_map_returns_none_if_missing(self, tmp_path):
        state = self._state(tmp_path)
        assert state.load_kc_map() is None

    def test_append_and_load_iteration_logs(self, tmp_path):
        from knowledge_miner import IterationLog

        state = self._state(tmp_path)
        log = IterationLog(
            iteration=0,
            timestamp="2026-01-01T00:00:00+00:00",
            subset_lo_ids=["lo-001"],
            subset_kc_ids=["kc-a"],
            kcs_added=1,
            kcs_modified=0,
            kcs_removed=0,
            improvements=["add more KCs"],
            rationale="test",
            quality_score_before=0.5,
            quality_score_after=0.6,
        )
        state.append_iteration_log(log)
        state.append_iteration_log(log)
        logs = state.load_iteration_logs()
        assert len(logs) == 2

    def test_save_and_load_quality_report(self, tmp_path):
        state = self._state(tmp_path)
        report = QualityReport(
            iteration=1,
            overall_score=0.9,
            coverage_score=1.0,
            granularity_score=0.8,
            distinctiveness_score=0.9,
            completeness_score=0.9,
            total_kcs=5,
            total_los=3,
        )
        state.save_quality_report(report)
        loaded = state.load_quality_report()
        assert loaded is not None
        assert loaded.overall_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Full Integration: KnowledgeMiner.run() with MockLLM
# ---------------------------------------------------------------------------


class TestKnowledgeMinerIntegration:
    def test_dry_run_produces_kc_map(self, tmp_path):
        state = RunState(tmp_path)
        llm = MockLLMClient()
        miner = KnowledgeMiner(
            llm=llm,
            state=state,
            max_iterations=2,
            batch_size=5,
            quality_threshold=0.99,  # high threshold so loop runs
            use_embeddings=False,
        )
        ci = get_default_course_input()
        state.save_course_input(ci)
        kc_map = miner.run(ci)
        assert len(kc_map.kcs) >= 17

    def test_state_files_created(self, tmp_path):
        state = RunState(tmp_path)
        llm = MockLLMClient()
        miner = KnowledgeMiner(
            llm=llm, state=state, max_iterations=1, batch_size=5,
            quality_threshold=0.99, use_embeddings=False,
        )
        ci = get_default_course_input()
        state.save_course_input(ci)
        miner.run(ci)
        assert state.kc_map_path.exists()
        assert state.quality_report_path.exists()
        assert state.iteration_logs_path.exists()

    def test_resume_restores_state(self, tmp_path):
        """Running twice with --resume should not lose KCs."""
        state = RunState(tmp_path)
        llm = MockLLMClient()
        miner = KnowledgeMiner(
            llm=llm, state=state, max_iterations=1, batch_size=5,
            quality_threshold=0.50, use_embeddings=False,
        )
        ci = get_default_course_input()
        state.save_course_input(ci)
        kc_map1 = miner.run(ci)

        # Second run resumes (mock produces same initial KCs so no new ones added)
        miner2 = KnowledgeMiner(
            llm=llm, state=state, max_iterations=1, batch_size=5,
            quality_threshold=0.50, use_embeddings=False,
        )
        # Load start iteration from existing logs
        logs = state.load_iteration_logs()
        start = max(log.iteration for log in logs) + 1 if logs else 0
        kc_map2 = miner2.run(ci, start_iteration=start)
        assert len(kc_map2.kcs) >= len(kc_map1.kcs)

    def test_quality_threshold_stops_loop(self, tmp_path):
        """With a very low threshold the loop should stop after iteration 0."""
        state = RunState(tmp_path)
        llm = MockLLMClient()
        miner = KnowledgeMiner(
            llm=llm, state=state, max_iterations=10, batch_size=5,
            quality_threshold=0.01, use_embeddings=False,
        )
        ci = get_default_course_input()
        state.save_course_input(ci)
        miner.run(ci)
        logs = state.load_iteration_logs()
        # Should stop early, not run all 10 iterations
        assert len(logs) <= 2
