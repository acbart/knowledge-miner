#!/usr/bin/env python3
"""
Knowledge Component (KC) Decomposition Tool for Intelligent Tutoring Systems.

Uses an LLM in an iterative loop to decompose course learning objectives into
exhaustive, well-structured knowledge components suitable for an ITS.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ID Generation
# ---------------------------------------------------------------------------


def make_lo_id(index: int) -> str:
    return f"lo-{index:03d}"


def make_kc_id(title: str, existing_ids: set) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    candidate = f"kc-{slug}"
    if candidate not in existing_ids:
        return candidate
    i = 2
    while f"{candidate}-{i}" in existing_ids:
        i += 1
    return f"{candidate}-{i}"


def normalize_lo_ids(objectives: List[LearningObjective]) -> List[LearningObjective]:
    return [
        LearningObjective(id=make_lo_id(i), title=lo.title, description=lo.description)
        for i, lo in enumerate(objectives, start=1)
    ]


# ---------------------------------------------------------------------------
# JSON Parsing & Repair
# ---------------------------------------------------------------------------


class JSONParseError(Exception):
    pass


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```$", "", text)
    return text.strip()


def _repair_pass1(text: str) -> str:
    """Remove trailing commas before } or ]."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _repair_pass2(text: str) -> str:
    """Pass 1 + naive single-quote → double-quote replacement."""
    text = _repair_pass1(text)
    text = re.sub(r"'([^'\\]*)'", r'"\1"', text)
    return text


def _repair_pass3(text: str) -> str:
    """Pass 2 + truncate at last valid closing delimiter."""
    text = _repair_pass2(text)
    last = max(text.rfind("}"), text.rfind("]"))
    if last >= 0:
        text = text[: last + 1]
    return text


def parse_json_with_repair(text: str) -> Any:
    """Try to parse JSON, applying progressively more aggressive repairs on failure."""
    text = _strip_code_fences(text)
    for repair_fn in (None, _repair_pass1, _repair_pass2, _repair_pass3):
        candidate = repair_fn(text) if repair_fn else text
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise JSONParseError(f"Failed to parse JSON after all repair attempts. Preview: {text[:200]!r}")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.debug("Saved JSON → %s", path)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Deduplication Logic
# ---------------------------------------------------------------------------


def compute_title_similarity(t1: str, t2: str) -> float:
    return difflib.SequenceMatcher(None, t1.lower(), t2.lower()).ratio()


def find_duplicate_kcs(
    kcs: List[KnowledgeComponent], threshold: float = 0.85
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i, a in enumerate(kcs):
        for b in kcs[i + 1 :]:
            if compute_title_similarity(a.title, b.title) >= threshold:
                pairs.append((a.id, b.id))
    return pairs


class EmbeddingSimilarity:
    """Optional numpy-based bag-of-words cosine similarity."""

    def __init__(self) -> None:
        try:
            import numpy as np  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def compute(self, texts: List[str]) -> Any:
        if not self._available:
            raise RuntimeError("numpy is not available")
        import numpy as np

        vocab: Dict[str, int] = {}
        for text in texts:
            for word in text.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        mat = np.zeros((len(texts), len(vocab)), dtype=float)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                mat[i, vocab[word]] += 1
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


# ---------------------------------------------------------------------------
# LLM Client Abstraction
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required: pip install openai")
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307") -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package is required: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return message.content[0].text


class MockLLMClient(LLMClient):
    """Smart mock that detects operation type from prompt content and returns valid JSON."""

    # --- Static mock payloads ---

    _MOCK_KCS: List[dict] = [
        {
            "id": "kc-variable-assignment",
            "title": "Variable Assignment",
            "description": "Variables are names bound to values; assignment updates that binding.",
            "parent_lo_ids": ["lo-001"],
            "prerequisites": [],
            "type": "concept",
            "granularity_level": "fine",
            "examples": ["x = 5", "name = 'Alice'", "total = a + b"],
            "non_examples": ["5 = x  # invalid", "print(x)  # not assignment"],
            "common_misconceptions": [
                "Variables store the expression, not the evaluated value",
                "x = x + 1 is a contradiction like in math",
            ],
            "observable_evidence": [
                "Correctly assigns values and retrieves them",
                "Distinguishes = (assignment) from == (equality)",
            ],
            "likely_errors": [
                "Using = instead of == in conditions",
                "Referencing an undefined variable",
            ],
            "practice_tasks": [
                "Trace variable values through a sequence of assignments",
                "Predict output after multi-step assignments",
            ],
            "assessment_cues": ["What does x = x + 1 do? What is x after x = 3; x = x * 2?"],
            "relationships": [],
        },
        {
            "id": "kc-data-types-basic",
            "title": "Basic Data Types (int, float, str, bool, None)",
            "description": "Python's five fundamental scalar types and their properties.",
            "parent_lo_ids": ["lo-001"],
            "prerequisites": [],
            "type": "concept",
            "granularity_level": "fine",
            "examples": ["42 (int)", "3.14 (float)", "'hello' (str)", "True (bool)", "None"],
            "non_examples": ["[1, 2, 3]  # list, not a scalar", "{} # dict"],
            "common_misconceptions": [
                "'5' and 5 are the same value",
                "bool is unrelated to int in Python",
            ],
            "observable_evidence": [
                "Uses type() to inspect types correctly",
                "Avoids mixing incompatible types without conversion",
            ],
            "likely_errors": [
                "Concatenating str and int without str()",
                "Expecting integer division from /",
            ],
            "practice_tasks": [
                "Identify the type of a dozen given literals",
                "Fix a TypeError caused by mixing str and int",
            ],
            "assessment_cues": ["What is type(True)? What does 3 + '3' produce?"],
            "relationships": [],
        },
        {
            "id": "kc-arithmetic-expressions",
            "title": "Arithmetic Expressions and Operators",
            "description": "Forming and evaluating expressions with +, -, *, /, //, %, **.",
            "parent_lo_ids": ["lo-002"],
            "prerequisites": ["kc-data-types-basic"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": ["2 + 3", "10 // 3", "2 ** 8", "7 % 3"],
            "non_examples": ["'a' + 'b'  # string concat, not arithmetic"],
            "common_misconceptions": [
                "/ always returns an int",
                "% is percentage, not remainder",
            ],
            "observable_evidence": [
                "Correctly evaluates arithmetic expressions",
                "Applies operator precedence (PEMDAS)",
            ],
            "likely_errors": [
                "Confusing / and // for integer division",
                "Wrong precedence without parentheses",
            ],
            "practice_tasks": [
                "Evaluate expressions by hand",
                "Write an expression to compute the area of a circle",
            ],
            "assessment_cues": ["What is 7 // 2? What is 7 % 3? What is 2 ** 10?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-data-types-basic"}],
        },
        {
            "id": "kc-boolean-expressions",
            "title": "Boolean Expressions and Logical Operators",
            "description": "Constructing and evaluating boolean expressions with comparison and logical operators.",
            "parent_lo_ids": ["lo-002", "lo-003"],
            "prerequisites": ["kc-data-types-basic"],
            "type": "concept",
            "granularity_level": "fine",
            "examples": ["x > 0", "a == b", "not flag", "x > 0 and x < 10"],
            "non_examples": ["x = 5  # assignment, not comparison"],
            "common_misconceptions": [
                "and is the same as bitwise &",
                "not x > 0 always means x <= 0",
            ],
            "observable_evidence": [
                "Correctly uses ==, !=, <, >, and, or, not",
                "Understands short-circuit evaluation",
            ],
            "likely_errors": [
                "Using = instead of ==",
                "Operator precedence errors in compound conditions",
            ],
            "practice_tasks": [
                "Evaluate boolean expressions by hand",
                "Write conditions for a grading scenario",
            ],
            "assessment_cues": ["What does (True or False and False) evaluate to? Why?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-data-types-basic"}],
        },
        {
            "id": "kc-conditional-branching",
            "title": "Conditional Branching (if/elif/else)",
            "description": "Using if, elif, else to control program flow based on boolean conditions.",
            "parent_lo_ids": ["lo-003"],
            "prerequisites": ["kc-boolean-expressions", "kc-variable-assignment"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": ["if x > 0: ...", "if/else for binary choice", "elif chains"],
            "non_examples": ["while loop  # iteration, not branching"],
            "common_misconceptions": [
                "elif still runs when the preceding if was True",
                "Missing else causes an error",
            ],
            "observable_evidence": [
                "Writes correct if/elif/else for multi-way decisions",
                "Handles all edge cases including equality",
            ],
            "likely_errors": [
                "Forgetting the colon",
                "Using assignment = instead of comparison ==",
                "Off-by-one boundary conditions",
            ],
            "practice_tasks": [
                "Write a letter-grade classifier",
                "Trace output through nested conditionals",
            ],
            "assessment_cues": ["Can the student predict output of an if/elif/else chain?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-boolean-expressions"}],
        },
        {
            "id": "kc-for-loop-iteration",
            "title": "For Loop and range()",
            "description": "Using for loops to iterate over sequences and ranges.",
            "parent_lo_ids": ["lo-004"],
            "prerequisites": ["kc-variable-assignment", "kc-data-types-basic"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": ["for i in range(10):", "for item in my_list:", "for ch in s:"],
            "non_examples": ["while loop  # condition-controlled"],
            "common_misconceptions": [
                "range(n) includes n",
                "Modifying a list while iterating is safe",
            ],
            "observable_evidence": [
                "Writes correct for loops over lists, ranges, and strings",
                "Uses range() with start/stop/step correctly",
            ],
            "likely_errors": [
                "Off-by-one with range()",
                "Forgetting colon",
                "Wrong indentation of loop body",
            ],
            "practice_tasks": [
                "Sum a list of numbers",
                "Print a multiplication table",
                "Count occurrences of a character in a string",
            ],
            "assessment_cues": ["How many times does range(3, 8, 2) iterate?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-variable-assignment"}],
        },
        {
            "id": "kc-while-loop",
            "title": "While Loop and Loop Control (break, continue)",
            "description": "Condition-controlled repetition and loop control statements.",
            "parent_lo_ids": ["lo-004"],
            "prerequisites": ["kc-boolean-expressions", "kc-variable-assignment"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": ["while x > 0: x -= 1", "while True: ... break"],
            "non_examples": ["for loop  # count-controlled"],
            "common_misconceptions": [
                "An infinite loop is always a bug",
                "break exits the whole program",
            ],
            "observable_evidence": [
                "Writes while loops with valid termination conditions",
                "Uses break/continue appropriately",
            ],
            "likely_errors": [
                "Infinite loop due to missing update",
                "break/continue used outside a loop",
            ],
            "practice_tasks": [
                "Write an input-validation loop",
                "Implement a number-guessing game with while",
            ],
            "assessment_cues": ["When would you use while instead of for?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-boolean-expressions"}],
        },
        {
            "id": "kc-function-definition",
            "title": "Function Definition and Calling",
            "description": "Defining functions with def, parameters, and return values; calling them correctly.",
            "parent_lo_ids": ["lo-005"],
            "prerequisites": ["kc-variable-assignment", "kc-data-types-basic"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": ["def add(a, b): return a + b", "result = add(3, 4)", "def greet(name='World'):"],
            "non_examples": ["lambda  # anonymous function", "calling without parentheses"],
            "common_misconceptions": [
                "A function runs when it is defined",
                "return sends output directly to print()",
            ],
            "observable_evidence": [
                "Defines functions with correct syntax",
                "Uses return values rather than printing inside",
            ],
            "likely_errors": [
                "Missing return statement",
                "Wrong number of positional arguments",
                "Reusing function name as a variable",
            ],
            "practice_tasks": [
                "Write a function that checks whether a number is even",
                "Refactor repeated code into a helper function",
            ],
            "assessment_cues": ["What does a function return when there is no return statement?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-variable-assignment"}],
        },
        {
            "id": "kc-list-operations",
            "title": "List Creation, Indexing, and Methods",
            "description": "Creating lists, indexing, slicing, appending, and using built-in list methods.",
            "parent_lo_ids": ["lo-006"],
            "prerequisites": ["kc-variable-assignment", "kc-data-types-basic"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": ["[1, 2, 3]", "lst.append(4)", "lst[0]", "lst[1:3]", "len(lst)"],
            "non_examples": ["Tuple  # immutable sequence", "Dict  # key-value"],
            "common_misconceptions": [
                "Negative indices always raise IndexError",
                "append() returns a new list",
            ],
            "observable_evidence": [
                "Correctly indexes, slices, and modifies lists",
                "Uses list methods (append, pop, sort) without confusion",
            ],
            "likely_errors": [
                "IndexError from out-of-range index",
                "Mutating a list while iterating over it",
            ],
            "practice_tasks": [
                "Build a shopping-list program",
                "Reverse a list without using .reverse()",
            ],
            "assessment_cues": ["What is [10, 20, 30][-1]? What does lst.append(5) return?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-data-types-basic"}],
        },
        {
            "id": "kc-string-methods",
            "title": "String Methods and f-String Formatting",
            "description": "Common string methods and f-string syntax for constructing output.",
            "parent_lo_ids": ["lo-006"],
            "prerequisites": ["kc-data-types-basic"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": ["'hello'.upper()", "'a,b,c'.split(',')", "f'Hello {name}'", "'  hi  '.strip()"],
            "non_examples": ["List methods  # different object type"],
            "common_misconceptions": [
                "Strings are mutable like lists",
                "str.replace() modifies the string in place",
            ],
            "observable_evidence": [
                "Applies string methods correctly and chains them",
                "Constructs formatted output with f-strings",
            ],
            "likely_errors": [
                "Calling a list method on a string",
                "Forgetting the f prefix in f-strings",
            ],
            "practice_tasks": [
                "Parse a comma-separated line from user input",
                "Format a receipt using f-strings",
            ],
            "assessment_cues": ["What does 'Hello World'.split() return? Is it mutable?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-data-types-basic"}],
        },
        {
            "id": "kc-dict-operations",
            "title": "Dictionary Creation and Operations",
            "description": "Creating dicts, accessing by key, iterating, and using dict methods.",
            "parent_lo_ids": ["lo-007"],
            "prerequisites": ["kc-variable-assignment", "kc-data-types-basic"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": ["{'a': 1}", "d['key']", "d.get('key', default)", "for k, v in d.items():"],
            "non_examples": ["List  # indexed by position not key"],
            "common_misconceptions": [
                "Dict access with a missing key returns None",
                "Dicts preserve insertion order in Python 2",
            ],
            "observable_evidence": [
                "Correctly creates, reads, and updates dictionaries",
                "Uses .get() to avoid KeyError",
            ],
            "likely_errors": ["KeyError on missing key", "Iterating over keys when values needed"],
            "practice_tasks": [
                "Count word frequencies in a sentence",
                "Store and retrieve student grades by name",
            ],
            "assessment_cues": ["What happens when you access d['missing'] vs d.get('missing')?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-data-types-basic"}],
        },
        {
            "id": "kc-file-io",
            "title": "File Reading and Writing",
            "description": "Opening, reading, and writing text files using open() and context managers.",
            "parent_lo_ids": ["lo-008"],
            "prerequisites": ["kc-variable-assignment", "kc-string-methods"],
            "type": "tool_practical",
            "granularity_level": "fine",
            "examples": [
                "with open('f.txt') as f: data = f.read()",
                "f.readlines()",
                "with open('out.txt', 'w') as f: f.write(text)",
            ],
            "non_examples": ["Reading from a URL", "Binary file I/O"],
            "common_misconceptions": [
                "Files are automatically closed when the variable goes out of scope",
                "readlines() strips newlines",
            ],
            "observable_evidence": [
                "Opens files with context manager correctly",
                "Iterates over file lines and strips whitespace",
            ],
            "likely_errors": [
                "FileNotFoundError from wrong path",
                "Forgetting to strip newlines from readlines()",
            ],
            "practice_tasks": [
                "Read a CSV file and compute the sum of a column",
                "Write a list of items to a file, one per line",
            ],
            "assessment_cues": ["Why use 'with open(...)' instead of open() without with?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-string-methods"}],
        },
        {
            "id": "kc-debugging-runtime-errors",
            "title": "Reading Tracebacks and Debugging Runtime Errors",
            "description": "Interpreting Python tracebacks, identifying common error types, and locating bugs.",
            "parent_lo_ids": ["lo-009"],
            "prerequisites": ["kc-data-types-basic", "kc-variable-assignment"],
            "type": "debugging_skill",
            "granularity_level": "fine",
            "examples": [
                "Reading a NameError traceback",
                "Finding a TypeError in a function call",
                "Identifying which line caused an IndexError",
            ],
            "non_examples": [
                "SyntaxError  # caught at parse time, different workflow",
                "Logic errors  # no exception raised",
            ],
            "common_misconceptions": [
                "The error always originates at the line number shown",
                "All errors crash the entire program permanently",
            ],
            "observable_evidence": [
                "Reads and interprets tracebacks correctly",
                "Identifies error type and root cause",
            ],
            "likely_errors": [
                "Misreading the line number in nested calls",
                "Ignoring the 'During handling...' inner cause",
            ],
            "practice_tasks": [
                "Given a traceback, explain the cause and fix",
                "Distinguish NameError vs AttributeError vs TypeError",
            ],
            "assessment_cues": ["Given this traceback, what caused the error and on which line?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-variable-assignment"}],
        },
        {
            "id": "kc-indentation-blocks",
            "title": "Indentation and Code Block Structure",
            "description": "Python uses indentation (not braces) to define blocks; mixing tabs/spaces is illegal.",
            "parent_lo_ids": ["lo-001"],
            "prerequisites": [],
            "type": "concept",
            "granularity_level": "atomic",
            "examples": ["if x:\\n    do_thing()", "def f():\\n    return 1"],
            "non_examples": ["Using {} for blocks (Java/C style)"],
            "common_misconceptions": [
                "Any consistent indentation amount is fine within a file",
                "Tabs and spaces are interchangeable",
            ],
            "observable_evidence": [
                "Correctly indents all blocks",
                "Fixes IndentationError without hints",
            ],
            "likely_errors": [
                "Mixed tabs and spaces",
                "Incorrect nesting level",
                "Missing colon before a block",
            ],
            "practice_tasks": [
                "Fix indentation errors in given buggy code",
                "Draw the block structure of a nested if/for",
            ],
            "assessment_cues": ["What happens when you mix 2-space and 4-space indentation?"],
            "relationships": [],
        },
        {
            "id": "kc-print-input",
            "title": "print() and input() Built-in Functions",
            "description": "Using print() for output and input() for reading user input (always returns str).",
            "parent_lo_ids": ["lo-001"],
            "prerequisites": ["kc-data-types-basic"],
            "type": "tool_practical",
            "granularity_level": "atomic",
            "examples": ["print('Hello')", "name = input('Name: ')", "n = int(input('Enter n: '))"],
            "non_examples": ["sys.stdout.write()  # lower-level alternative"],
            "common_misconceptions": [
                "input() returns an int if you type a number",
                "print() returns the value it printed",
            ],
            "observable_evidence": [
                "Correctly uses input() and converts to needed type",
                "Avoids spurious print() calls in graded output",
            ],
            "likely_errors": [
                "Forgetting to convert input() to int or float",
                "Passing multiple arguments when sep is needed",
            ],
            "practice_tasks": [
                "Read a user's name and age and print a sentence",
                "Build a simple calculator with two input() calls",
            ],
            "assessment_cues": ["What type does input() always return? How do you get an integer from it?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-data-types-basic"}],
        },
        {
            "id": "kc-scope-namespaces",
            "title": "Variable Scope and Namespaces",
            "description": "Local vs. global scope; how functions create isolated namespaces.",
            "parent_lo_ids": ["lo-005"],
            "prerequisites": ["kc-variable-assignment", "kc-function-definition"],
            "type": "concept",
            "granularity_level": "medium",
            "examples": [
                "Local variable unreachable outside its function",
                "global keyword to write to a global",
            ],
            "non_examples": ["Class attributes  # different scoping rules"],
            "common_misconceptions": [
                "Variables inside functions affect global state by default",
                "global is needed just to read a global variable",
            ],
            "observable_evidence": [
                "Predicts which variable is accessed in nested scope",
                "Avoids unintended global mutation",
            ],
            "likely_errors": [
                "UnboundLocalError from referencing global before local assignment",
                "Accidental shadowing of a built-in name",
            ],
            "practice_tasks": [
                "Trace variable values through nested function calls",
                "Identify scope bugs in provided code",
            ],
            "assessment_cues": ["Can the student predict which x is printed in a LEGB example?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-function-definition"}],
        },
        {
            "id": "kc-program-decomposition",
            "title": "Program Design and Top-Down Decomposition",
            "description": "Breaking a problem into sub-problems, identifying reusable functions, writing docstrings.",
            "parent_lo_ids": ["lo-010"],
            "prerequisites": ["kc-function-definition", "kc-scope-namespaces"],
            "type": "strategy",
            "granularity_level": "medium",
            "examples": [
                "Drawing a call graph before coding",
                "Writing docstrings before implementing",
                "DRY principle: extract repeated logic",
            ],
            "non_examples": ["Writing one giant main() function", "Skipping planning"],
            "common_misconceptions": [
                "Decomposition is only for large programs",
                "More functions always mean more complexity",
            ],
            "observable_evidence": [
                "Produces code with well-named, single-purpose functions",
                "Writes docstrings for all public functions",
            ],
            "likely_errors": [
                "Functions that do too many unrelated things",
                "Hardcoded values that should be parameters",
            ],
            "practice_tasks": [
                "Decompose a word-count program into helper functions",
                "Critique a monolithic solution and refactor it",
            ],
            "assessment_cues": ["Given a 50-line main(), identify candidates for extraction."],
            "relationships": [{"type": "assumes", "kc_id": "kc-function-definition"}],
        },
    ]

    _CRITIQUE_RESPONSE: dict = {
        "issues": [
            "KCs for sets (lo-007) are missing",
            "Exception handling (try/except) is not yet a KC",
            "observable_evidence could be richer for procedure-type KCs",
        ],
        "recommendations": [
            "Add a KC for set creation and membership testing",
            "Add a KC for try/except exception handling",
            "Expand observable_evidence with measurable verbs for procedure KCs",
        ],
        "coverage_score": 0.70,
        "granularity_score": 0.82,
        "distinctiveness_score": 0.91,
    }

    _EXTRA_KCS: List[dict] = [
        {
            "id": "kc-set-operations",
            "title": "Set Creation and Membership Testing",
            "description": "Creating sets, adding/removing elements, and testing membership efficiently.",
            "parent_lo_ids": ["lo-007"],
            "prerequisites": ["kc-data-types-basic"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": ["{1, 2, 3}", "s.add(4)", "3 in s", "a | b  # union"],
            "non_examples": ["Dict  # uses {} but is key-value"],
            "common_misconceptions": [
                "Sets are ordered",
                "{} creates an empty set (it creates an empty dict)",
            ],
            "observable_evidence": [
                "Creates sets correctly including empty set()",
                "Uses in for O(1) membership check instead of a list",
            ],
            "likely_errors": [
                "Using {} for empty set",
                "Attempting to index into a set",
            ],
            "practice_tasks": [
                "Find unique words in a sentence",
                "Compute the intersection of two groups",
            ],
            "assessment_cues": ["How do you create an empty set? What is {1,2} | {2,3}?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-data-types-basic"}],
        },
        {
            "id": "kc-exception-handling",
            "title": "Exception Handling with try/except",
            "description": "Using try/except blocks to catch and handle runtime exceptions gracefully.",
            "parent_lo_ids": ["lo-009"],
            "prerequisites": ["kc-debugging-runtime-errors"],
            "type": "procedure",
            "granularity_level": "fine",
            "examples": [
                "try: int(x)\nexcept ValueError: ...",
                "except (TypeError, ValueError) as e:",
                "finally: cleanup()",
            ],
            "non_examples": [
                "Bare except:  # catches everything silently",
                "Silently swallowing all exceptions",
            ],
            "common_misconceptions": [
                "try/except makes code noticeably slower",
                "except Exception catches KeyboardInterrupt",
            ],
            "observable_evidence": [
                "Catches specific exception types",
                "Avoids bare except; logs or re-raises unexpected errors",
            ],
            "likely_errors": [
                "Bare except clause",
                "Catching too broadly with except Exception",
                "Not re-raising unexpected errors",
            ],
            "practice_tasks": [
                "Validate integer input from the user with try/except",
                "Open a file safely and handle FileNotFoundError",
            ],
            "assessment_cues": ["What is the difference between 'except Exception' and bare 'except:'?"],
            "relationships": [{"type": "assumes", "kc_id": "kc-debugging-runtime-errors"}],
        },
    ]

    def _detect_operation(self, system_prompt: str, user_prompt: str) -> str:
        """Detect the type of LLM call from the (controlled) system prompt content."""
        # Check system prompt first since we control its content precisely
        if "from_kc_id" in system_prompt:
            return "relationship"
        if "discard_id" in system_prompt:
            return "deduplication"
        if "coverage_score" in system_prompt:
            return "critique"
        if '"added"' in system_prompt and '"modified"' in system_prompt:
            return "refinement"
        return "generation"

    def complete(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.2
    ) -> str:
        op = self._detect_operation(system_prompt, user_prompt)
        logger.debug("MockLLM operation detected: %s", op)

        if op == "generation":
            return json.dumps(self._MOCK_KCS)
        if op == "critique":
            return json.dumps(self._CRITIQUE_RESPONSE)
        if op == "refinement":
            return json.dumps(
                {
                    "added": self._EXTRA_KCS,
                    "modified": [],
                    "removed": [],
                    "rationale": (
                        "Added set-operations KC (lo-007 was uncovered) "
                        "and exception-handling KC identified in critique."
                    ),
                }
            )
        if op == "relationship":
            return json.dumps(
                {
                    "relationships": [
                        {
                            "from_kc_id": "kc-conditional-branching",
                            "type": "assumes",
                            "kc_id": "kc-boolean-expressions",
                        },
                        {
                            "from_kc_id": "kc-scope-namespaces",
                            "type": "assumes",
                            "kc_id": "kc-function-definition",
                        },
                        {
                            "from_kc_id": "kc-exception-handling",
                            "type": "assumes",
                            "kc_id": "kc-debugging-runtime-errors",
                        },
                        {
                            "from_kc_id": "kc-program-decomposition",
                            "type": "extends",
                            "kc_id": "kc-function-definition",
                        },
                    ]
                }
            )
        if op == "deduplication":
            return json.dumps({"merges": []})
        return json.dumps([])


# ---------------------------------------------------------------------------
# Prompt Templates
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


def build_relationship_prompt(kcs: List[KnowledgeComponent]) -> str:
    summary = json.dumps(
        [{"id": kc.id, "title": kc.title, "description": kc.description[:120]} for kc in kcs],
        indent=2,
    )
    return f"Knowledge Components:\n{summary}\n\nIdentify relationships. Return JSON."


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


def build_deduplication_prompt(
    kcs: List[KnowledgeComponent], pairs: List[Tuple[str, str]]
) -> str:
    kc_by_id = {kc.id: kc for kc in kcs}
    pair_data = [
        {"kc1": kc_by_id[a].model_dump(), "kc2": kc_by_id[b].model_dump()}
        for a, b in pairs
        if a in kc_by_id and b in kc_by_id
    ]
    return f"Potentially duplicate KC pairs:\n{json.dumps(pair_data, indent=2)}\n\nDecide merges. Return JSON."


# ---------------------------------------------------------------------------
# Relationship Validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# KC Map Operations
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Batch Selection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Default CS1 Course Input
# ---------------------------------------------------------------------------


def get_default_course_input() -> CourseInput:
    raw = [
        LearningObjective(
            id="",
            title="Variables, Data Types, and Assignment",
            description=(
                "Students can declare variables, assign values, and use Python's "
                "basic data types (int, float, str, bool, None)."
            ),
        ),
        LearningObjective(
            id="",
            title="Expressions and Operators",
            description=(
                "Students can form and evaluate arithmetic, string, and boolean "
                "expressions using Python operators."
            ),
        ),
        LearningObjective(
            id="",
            title="Conditionals",
            description=(
                "Students can write if/elif/else statements to control program "
                "flow based on conditions."
            ),
        ),
        LearningObjective(
            id="",
            title="Loops",
            description=(
                "Students can write for and while loops, use range(), and apply "
                "loop control (break, continue)."
            ),
        ),
        LearningObjective(
            id="",
            title="Functions",
            description=(
                "Students can define and call functions with parameters, default "
                "values, and return values."
            ),
        ),
        LearningObjective(
            id="",
            title="Lists and Strings",
            description=(
                "Students can create and manipulate lists and strings using "
                "indexing, slicing, and built-in methods."
            ),
        ),
        LearningObjective(
            id="",
            title="Dictionaries and Sets",
            description=(
                "Students can create and use dictionaries and sets for key-value "
                "storage and membership testing."
            ),
        ),
        LearningObjective(
            id="",
            title="File I/O",
            description=(
                "Students can read from and write to text files using open(), "
                "read(), write(), and context managers."
            ),
        ),
        LearningObjective(
            id="",
            title="Debugging and Error Handling",
            description=(
                "Students can read tracebacks, identify common error types, and "
                "use try/except for exception handling."
            ),
        ),
        LearningObjective(
            id="",
            title="Program Design and Decomposition",
            description=(
                "Students can design programs using top-down decomposition, "
                "identify reusable functions, and write docstrings."
            ),
        ),
    ]
    return CourseInput(
        course_description="CS1: Introductory Programming in Python",
        learning_objectives=normalize_lo_ids(raw),
    )


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------


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
        logger.info("Running initial KC generation for %d LOs...", len(course_input.learning_objectives))
        raw_kcs = self._generate(course_input.course_description, course_input.learning_objectives, [])

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

            subset_los, subset_kcs = select_subset(los, kc_map.kcs, self.batch_size, iteration)
            logger.debug("Subset: %d LOs, %d KCs", len(subset_los), len(subset_kcs))

            # Critique
            try:
                critique = self._critique(course_input.course_description, subset_los, kc_map.kcs)
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
            logger.info("Critique score: %.3f (threshold: %.2f)", critique_score, self.quality_threshold)

            if critique_score >= self.quality_threshold:
                logger.info("Quality threshold reached via critique. Stopping.")
                self.state.save_quality_report(
                    compute_quality_score(
                        kc_map.kcs, los, iteration,
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
                refinement = {"added": [], "modified": [], "removed": [], "rationale": "failed"}

            existing_ids = {kc.id for kc in kc_map.kcs}
            kc_map, kcs_added, kcs_modified, kcs_removed = apply_refinement(
                kc_map, refinement, existing_ids
            )
            logger.info(
                "Changes: +%d added, ~%d modified, -%d removed", kcs_added, kcs_modified, kcs_removed
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
                kc_map.kcs, los, iteration,
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


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------


def create_llm_client(
    provider: str, model: Optional[str], api_key: Optional[str]
) -> LLMClient:
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OpenAI API key required (--api-key or OPENAI_API_KEY env var)")
        return OpenAIClient(api_key=key, model=model or "gpt-4o-mini")
    if provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("Anthropic API key required (--api-key or ANTHROPIC_API_KEY env var)")
        return AnthropicClient(api_key=key, model=model or "claude-3-haiku-20240307")
    raise ValueError(f"Unknown LLM provider: {provider!r}. Choose openai, anthropic, or mock.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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


if __name__ == "__main__":
    main()
