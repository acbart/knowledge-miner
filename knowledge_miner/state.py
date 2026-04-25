"""Persistent run state: saving and loading KC maps, course inputs, logs, and reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .json_utils import append_jsonl, load_json, load_jsonl, save_json
from .models import CourseInput, IterationLog, KCMap, QualityReport


class RunState:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.course_input_path = output_dir / "course_input.json"
        self.kc_map_path = output_dir / "kc_map.json"
        self.iteration_logs_path = output_dir / "iteration_logs.jsonl"
        self.quality_report_path = output_dir / "quality_report.json"

    def save_course_input(self, course_input: CourseInput) -> None:
        save_json(self.course_input_path, course_input.model_dump())

    def load_course_input(self) -> CourseInput:
        return CourseInput.model_validate(load_json(self.course_input_path))

    def save_kc_map(self, kc_map: KCMap) -> None:
        kc_map.last_updated = datetime.now(timezone.utc).isoformat()
        save_json(self.kc_map_path, kc_map.model_dump())

    def load_kc_map(self) -> Optional[KCMap]:
        if not self.kc_map_path.exists():
            return None
        return KCMap.model_validate(load_json(self.kc_map_path))

    def save_quality_report(self, report: QualityReport) -> None:
        save_json(self.quality_report_path, report.model_dump())

    def load_quality_report(self) -> Optional[QualityReport]:
        if not self.quality_report_path.exists():
            return None
        return QualityReport.model_validate(load_json(self.quality_report_path))

    def append_iteration_log(self, log: IterationLog) -> None:
        append_jsonl(self.iteration_logs_path, log.model_dump())

    def load_iteration_logs(self) -> List[IterationLog]:
        return [IterationLog.model_validate(r) for r in load_jsonl(self.iteration_logs_path)]
