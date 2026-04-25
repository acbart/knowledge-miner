"""LLM client abstractions: base class, OpenAI, Anthropic, and Mock implementations."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


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
            "examples": [
                "def add(a, b): return a + b",
                "result = add(3, 4)",
                "def greet(name='World'):",
            ],
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
            "assessment_cues": [
                "What does a function return when there is no return statement?"
            ],
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
            "assessment_cues": [
                "What is [10, 20, 30][-1]? What does lst.append(5) return?"
            ],
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
            "examples": [
                "'hello'.upper()",
                "'a,b,c'.split(',')",
                "f'Hello {name}'",
                "'  hi  '.strip()",
            ],
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
            "examples": [
                "{'a': 1}",
                "d['key']",
                "d.get('key', default)",
                "for k, v in d.items():",
            ],
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
            "assessment_cues": [
                "What happens when you access d['missing'] vs d.get('missing')?"
            ],
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
            "assessment_cues": [
                "Why use 'with open(...)' instead of open() without with?"
            ],
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
            "assessment_cues": [
                "Given this traceback, what caused the error and on which line?"
            ],
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
            "assessment_cues": [
                "What happens when you mix 2-space and 4-space indentation?"
            ],
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
            "examples": [
                "print('Hello')",
                "name = input('Name: ')",
                "n = int(input('Enter n: '))",
            ],
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
            "assessment_cues": [
                "What type does input() always return? How do you get an integer from it?"
            ],
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
            "assessment_cues": [
                "Can the student predict which x is printed in a LEGB example?"
            ],
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
            "assessment_cues": [
                "Given a 50-line main(), identify candidates for extraction."
            ],
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
            "assessment_cues": [
                "What is the difference between 'except Exception' and bare 'except:'?"
            ],
            "relationships": [
                {"type": "assumes", "kc_id": "kc-debugging-runtime-errors"}
            ],
        },
    ]

    def _detect_operation(self, system_prompt: str, user_prompt: str) -> str:
        """Detect the type of LLM call from the (controlled) system prompt content."""
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
