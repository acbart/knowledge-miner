"""Default CS1 course input for demonstration and testing."""

from __future__ import annotations

from .id_utils import normalize_lo_ids
from .models import CourseInput, LearningObjective


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
