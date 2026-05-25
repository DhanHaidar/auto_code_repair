from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass
class BugExample:
    """Container for a single bug example."""

    buggy_code: str
    error_message: str
    correct_code: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationRecord:
    """Holds evaluation details for one example."""

    index: int
    passed: bool
    prompt: str
    predicted_code: str
    example: BugExample


@dataclass
class EvaluationResult:
    """Aggregated evaluation results for one instruction."""

    instruction: str
    score: float
    records: List[EvaluationRecord]

    def failures(self) -> List[EvaluationRecord]:
        """Return the evaluation records that failed."""
        return [record for record in self.records if not record.passed]


@dataclass
class OptimizationResult:
    """Summary of the optimization process."""

    best_instruction: str
    best_score: float
    history: List[EvaluationResult]


def normalize_code(code: str) -> str:
    """Normalize code by collapsing whitespace for comparison."""
    return re.sub(r"\s+", " ", code.strip())


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase identifier-like tokens."""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


def vectorize(tokens: Iterable[str]) -> Dict[str, int]:
    """Convert tokens into a simple frequency map."""
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def cosine_similarity(vec_a: Dict[str, int], vec_b: Dict[str, int]) -> float:
    """Compute cosine similarity between two frequency maps."""
    if not vec_a or not vec_b:
        return 0.0
    shared = set(vec_a.keys()) | set(vec_b.keys())
    dot = sum(vec_a.get(key, 0) * vec_b.get(key, 0) for key in shared)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class DatasetLoader:
    """Load bug datasets from JSON or in-memory structures."""

    def load_from_json(self, path: str) -> List[BugExample]:
        """Load a dataset from a JSON file path."""
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return self.load_from_list(data)

    def load_from_list(self, items: Sequence[dict]) -> List[BugExample]:
        """Load a dataset from a list of dictionaries."""
        examples: List[BugExample] = []
        for item in items:
            examples.append(
                BugExample(
                    buggy_code=item["buggy_code"],
                    error_message=item.get("error_message", ""),
                    correct_code=item["correct_code"],
                    metadata=item.get("metadata", {}),
                )
            )
        return examples


class PromptSignature:
    """Render prompts using a template with dynamic variables."""

    DEFAULT_TEMPLATE = (
        "Instruction:\n{instruction}\n\n"
        "Few-shot examples:\n{few_shot_examples}\n\n"
        "Buggy code:\n{buggy_code}\n\n"
        "Error message:\n{error_message}\n\n"
        "Return only the fixed code."
    )

    def __init__(self, template: Optional[str] = None) -> None:
        """Create a prompt signature with an optional custom template."""
        self.template = template or self.DEFAULT_TEMPLATE

    def render(
        self,
        instruction: str,
        few_shot_examples: str,
        buggy_code: str,
        error_message: str,
        bug_description: str = "",
        feedback_section: str = "",
    ) -> str:
        """Render the final prompt string."""
        values = {
            "instruction": instruction.strip(),
            "few_shot_examples": few_shot_examples.strip() or "(none)",
            "buggy_code": buggy_code.strip(),
            "error_message": error_message.strip() or "(none)",
            "bug_description": bug_description.strip() or "",
            "feedback_section": feedback_section.strip() or "",
        }
        return self.template.format_map(values)


class ExampleSelector:
    """Select a small number of similar examples for few-shot prompting."""

    def __init__(self, max_examples: int = 2) -> None:
        """Initialize the selector with a maximum number of examples."""
        self.max_examples = max(0, int(max_examples))

    def select_examples(
        self, query: BugExample, corpus: Sequence[BugExample], k: Optional[int] = None
    ) -> List[BugExample]:
        """Select the top-k similar examples from the corpus."""
        limit = self.max_examples if k is None else max(0, int(k))
        if limit == 0:
            return []

        query_text = f"{query.buggy_code}\n{query.error_message}"
        query_vec = vectorize(tokenize(query_text))

        scored: List[tuple[float, BugExample]] = []
        for candidate in corpus:
            if candidate is query:
                continue
            candidate_text = f"{candidate.buggy_code}\n{candidate.error_message}"
            candidate_vec = vectorize(tokenize(candidate_text))
            similarity = cosine_similarity(query_vec, candidate_vec)
            scored.append((similarity, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def format_examples(self, examples: Sequence[BugExample]) -> str:
        """Format examples into a few-shot block for the prompt."""
        if not examples:
            return "(none)"
        blocks = []
        for idx, example in enumerate(examples, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"Example {idx}:",
                        "Buggy code:",
                        example.buggy_code.strip(),
                        "Error:",
                        example.error_message.strip() or "(none)",
                        "Fixed code:",
                        example.correct_code.strip(),
                    ]
                )
            )
        return "\n\n".join(blocks)


class LLMCodeGenerator:
    """Interface for LLM-based code generation."""

    def generate(self, prompt: str, buggy_code: str, error_message: str) -> str:
        """Generate a candidate fix from the prompt and bug context."""
        raise NotImplementedError


class LLMGeneratorAdapter(LLMCodeGenerator):
    """Adapter that lets the optimizer call an APR LLMGenerator."""

    def __init__(self, generator: Any) -> None:
        """Store the existing APR LLMGenerator instance."""
        self.generator = generator

    def generate(self, prompt: str, buggy_code: str, error_message: str) -> str:
        """Generate a fix using the provided prompt override."""
        return self.generator.generate(
            buggy_code=buggy_code,
            bug_description=error_message,
            prompt_override=prompt,
        )


class MockLLMCodeGenerator(LLMCodeGenerator):
    """Mock LLM that only fixes known bugs when the prompt is strong."""

    def __init__(self, fixes_by_signature: Dict[str, str]) -> None:
        """Initialize the mock generator with known fixes."""
        self.fixes_by_signature = fixes_by_signature
        self.strong_keywords = {
            "edge",
            "boundary",
            "test",
            "counterexample",
            "invariant",
            "reason",
            "analyze",
        }

    def generate(self, prompt: str, buggy_code: str, error_message: str) -> str:
        """Return a fix if the prompt appears sufficiently strong."""
        prompt_text = prompt.lower()
        should_fix = any(keyword in prompt_text for keyword in self.strong_keywords)
        signature = normalize_code(buggy_code)
        if should_fix and signature in self.fixes_by_signature:
            return self.fixes_by_signature[signature]
        return buggy_code


class InstructionLLMClient:
    """Interface for LLM clients used to propose new instructions."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a completion string from the LLM."""
        raise NotImplementedError


class OpenAIInstructionClient(InstructionLLMClient):
    """OpenAI-based client to generate instruction variations."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """Initialize the client with a model name."""
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the OpenAI API to get a completion."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if OpenAI is None or not api_key:
            raise RuntimeError("OpenAI client is not available or API key is missing.")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content or ""
        return content.strip()


class PromptOptimizer:
    """Generate new instruction candidates based on failures."""

    def __init__(self, llm_client: Optional[InstructionLLMClient] = None) -> None:
        """Initialize the optimizer with an optional LLM client."""
        self.llm_client = llm_client

    def propose_instructions(
        self,
        failures: Sequence[EvaluationRecord],
        current_instruction: str,
        num_variations: int = 3,
    ) -> List[str]:
        """Propose new instruction strings using LLM or heuristics."""
        if self.llm_client is not None:
            try:
                return self._propose_with_llm(
                    failures, current_instruction, num_variations
                )
            except Exception:
                return self._propose_with_heuristics(
                    failures, current_instruction, num_variations
                )
        return self._propose_with_heuristics(failures, current_instruction, num_variations)

    def _propose_with_llm(
        self,
        failures: Sequence[EvaluationRecord],
        current_instruction: str,
        num_variations: int,
    ) -> List[str]:
        """Use the LLM client to generate new instruction candidates."""
        failure_summary = self._summarize_failures(failures)
        system_prompt = (
            "You are an optimizer that rewrites prompt instructions for code repair."
        )
        user_prompt = (
            "Current instruction:\n"
            f"{current_instruction}\n\n"
            "Failure summary:\n"
            f"{failure_summary}\n\n"
            f"Provide {num_variations} improved instruction variants, one per line."
        )
        completion = self.llm_client.complete(system_prompt, user_prompt)
        candidates = [line.strip("- ") for line in completion.splitlines() if line.strip()]
        return candidates[:num_variations] or self._propose_with_heuristics(
            failures, current_instruction, num_variations
        )

    def _propose_with_heuristics(
        self,
        failures: Sequence[EvaluationRecord],
        current_instruction: str,
        num_variations: int,
    ) -> List[str]:
        """Generate heuristic instruction variations without external LLMs."""
        failure_summary = self._summarize_failures(failures)
        base = [
            (
                "Analyze the error message and buggy code, explain the root cause, "
                "then produce a minimal fix that targets boundary cases."
            ),
            (
                "Use the few-shot examples to infer the correct pattern, "
                "reason about tests, and return only the corrected code."
            ),
            (
                "Check invariants and edge cases, then propose a patch and "
                "validate it against the error message before returning code."
            ),
        ]
        if failure_summary:
            base = [
                f"{instruction} Focus on: {failure_summary}." for instruction in base
            ]
        if num_variations <= len(base):
            return base[:num_variations]
        while len(base) < num_variations:
            base.append(base[-1])
        return base

    def _summarize_failures(self, failures: Sequence[EvaluationRecord]) -> str:
        """Summarize failure patterns for prompt refinement."""
        if not failures:
            return ""
        messages = []
        for failure in failures[:3]:
            message = failure.example.error_message.strip()
            if message:
                messages.append(message)
        if not messages:
            return ""
        return "; ".join(messages)


class Evaluator:
    """Evaluate prompt instructions over a dataset using an LLM coder."""

    def __init__(
        self,
        coder: LLMCodeGenerator,
        prompt_signature: PromptSignature,
        example_selector: ExampleSelector,
    ) -> None:
        """Initialize the evaluator with coder, prompt template, and selector."""
        self.coder = coder
        self.prompt_signature = prompt_signature
        self.example_selector = example_selector

    def evaluate(
        self,
        dataset: Sequence[BugExample],
        instruction: str,
        few_shot_k: int = 2,
    ) -> EvaluationResult:
        """Evaluate an instruction across the dataset and return a score."""
        records: List[EvaluationRecord] = []
        for index, example in enumerate(dataset):
            selected = self.example_selector.select_examples(
                example, dataset, k=few_shot_k
            )
            few_shot_block = self.example_selector.format_examples(selected)
            bug_description = example.metadata.get("bug_description", "")
            feedback_section = example.metadata.get("feedback_section", "")
            prompt = self.prompt_signature.render(
                instruction,
                few_shot_block,
                example.buggy_code,
                example.error_message,
                bug_description=bug_description,
                feedback_section=feedback_section,
            )
            prediction = self.coder.generate(
                prompt=prompt,
                buggy_code=example.buggy_code,
                error_message=example.error_message,
            )
            passed = normalize_code(prediction) == normalize_code(example.correct_code)
            records.append(
                EvaluationRecord(
                    index=index,
                    passed=passed,
                    prompt=prompt,
                    predicted_code=prediction,
                    example=example,
                )
            )

        score = (
            sum(1 for record in records if record.passed) / len(records)
            if records
            else 0.0
        )
        return EvaluationResult(instruction=instruction, score=score, records=records)


class PromptOptimizationPipeline:
    """Main optimization loop for Automatic Prompt Optimization (APE)."""

    def __init__(
        self,
        evaluator: Evaluator,
        optimizer: PromptOptimizer,
        min_score: float = 1.0,
    ) -> None:
        """Initialize the pipeline with evaluator, optimizer, and target score."""
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.min_score = min_score

    def optimize_pipeline(
        self,
        dataset: Sequence[BugExample],
        initial_instruction: str,
        iterations: int = 3,
        few_shot_k: int = 2,
    ) -> OptimizationResult:
        """Run the optimization loop and return the best instruction found."""
        history: List[EvaluationResult] = []
        best_instruction = initial_instruction
        best_score = -1.0

        current_instruction = initial_instruction
        for _ in range(max(1, int(iterations))):
            result = self.evaluator.evaluate(
                dataset, current_instruction, few_shot_k=few_shot_k
            )
            history.append(result)
            if result.score > best_score:
                best_score = result.score
                best_instruction = result.instruction
            if result.score >= self.min_score:
                break

            failures = result.failures()
            candidates = self.optimizer.propose_instructions(
                failures, current_instruction
            )
            candidate_results = []
            for candidate in candidates:
                candidate_result = self.evaluator.evaluate(
                    dataset, candidate, few_shot_k=few_shot_k
                )
                history.append(candidate_result)
                candidate_results.append(candidate_result)

            if candidate_results:
                best_candidate = max(
                    candidate_results, key=lambda item: item.score
                )
                if best_candidate.score > best_score:
                    best_score = best_candidate.score
                    best_instruction = best_candidate.instruction
                    current_instruction = best_candidate.instruction

        return OptimizationResult(
            best_instruction=best_instruction,
            best_score=best_score,
            history=history,
        )


def build_mock_dataset() -> List[dict]:
    """Build a small mock dataset for quick testing."""
    return [
        {
            "buggy_code": """
            def add(a, b):
                return a - b
            """,
            "error_message": "Expected addition but got subtraction.",
            "correct_code": """
            def add(a, b):
                return a + b
            """,
        },
        {
            "buggy_code": """
            def is_even(n):
                return n % 2 == 1
            """,
            "error_message": "Even numbers should return True.",
            "correct_code": """
            def is_even(n):
                return n % 2 == 0
            """,
        },
        {
            "buggy_code": """
            def max_in_list(values):
                return min(values)
            """,
            "error_message": "Expected maximum value.",
            "correct_code": """
            def max_in_list(values):
                return max(values)
            """,
        },
    ]


def build_fix_map(examples: Sequence[BugExample]) -> Dict[str, str]:
    """Build a mapping from normalized buggy code to correct code."""
    mapping: Dict[str, str] = {}
    for example in examples:
        mapping[normalize_code(example.buggy_code)] = example.correct_code.strip()
    return mapping


def run_demo() -> None:
    """Run the APE pipeline with a mock dataset and print results."""
    loader = DatasetLoader()
    dataset = loader.load_from_list(build_mock_dataset())
    fix_map = build_fix_map(dataset)

    prompt_signature = PromptSignature()
    selector = ExampleSelector(max_examples=2)
    coder = MockLLMCodeGenerator(fix_map)
    evaluator = Evaluator(coder=coder, prompt_signature=prompt_signature, example_selector=selector)

    optimizer = PromptOptimizer()
    pipeline = PromptOptimizationPipeline(
        evaluator=evaluator,
        optimizer=optimizer,
        min_score=1.0,
    )

    initial_instruction = "Fix the bug and return the corrected code."
    result = pipeline.optimize_pipeline(
        dataset=dataset,
        initial_instruction=initial_instruction,
        iterations=2,
        few_shot_k=2,
    )

    print("Best instruction:")
    print(result.best_instruction)
    print("Best score:")
    print(f"{result.best_score:.2f}")


if __name__ == "__main__":
    run_demo()
