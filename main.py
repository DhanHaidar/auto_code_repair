from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ape_prompt_optimizer import (
    BugExample,
    DatasetLoader,
    Evaluator,
    ExampleSelector,
    LLMGeneratorAdapter,
    OpenAIInstructionClient,
    PromptOptimizationPipeline,
    PromptOptimizer,
    PromptSignature,
)
from batch_repair import BatchRepairRunner
from core.mcts import MCTS
from core.node import Node
from evaluator.tester import Tester
from llm.generator import LLMGenerator
from llm.prompt import APE_TEMPLATE, DEFAULT_INSTRUCTION
from storage.database import Database

LIGHT_MODEL_PRESET = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _detect_transformers_device() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        return -1

    return -1


def _resolve_model_name(model_name: str, preset: str) -> str:
    preset_value = preset.strip().lower()
    if preset_value == "light":
        return LIGHT_MODEL_PRESET
    return model_name


def _optimize_instruction_from_dataset(
    generator: LLMGenerator,
    dataset,
    initial_instruction: str,
    optimizer_model: str,
    iterations: int,
    min_score: float,
    few_shot_k: int,
) -> str:
    if not dataset:
        return initial_instruction

    selector = ExampleSelector(max_examples=few_shot_k)
    prompt_signature = PromptSignature(template=APE_TEMPLATE)
    coder = LLMGeneratorAdapter(generator)
    evaluator = Evaluator(
        coder=coder,
        prompt_signature=prompt_signature,
        example_selector=selector,
    )

    optimizer_client = None
    optimizer_model = optimizer_model.strip()
    if optimizer_model:
        optimizer_client = OpenAIInstructionClient(model=optimizer_model)
    optimizer = PromptOptimizer(optimizer_client)

    pipeline = PromptOptimizationPipeline(
        evaluator=evaluator,
        optimizer=optimizer,
        min_score=min_score,
    )

    result = pipeline.optimize_pipeline(
        dataset=dataset,
        initial_instruction=initial_instruction,
        iterations=iterations,
        few_shot_k=few_shot_k,
    )
    return result.best_instruction


def main() -> None:
    """Entry point for the APR tool."""
    parser = argparse.ArgumentParser(description="APR tool with MCTS + LLM")
    parser.add_argument("--code", default="", help="Path to buggy code file")
    parser.add_argument(
        "--test",
        "--tests",
        dest="tests",
        default="",
        help="Path to pytest tests",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--model-preset",
        default="",
        help="Use a local model preset: light",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "openai", "openrouter", "transformers"],
    )
    parser.add_argument("--bug-description", default="")
    parser.add_argument("--storage", default="apr_results.json")
    parser.add_argument("--module-name", default="")
    parser.add_argument("--package-name", default="")
    parser.add_argument("--max-patch-chars", type=int, default=8000)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--early-stop", dest="early_stop", action="store_true")
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--allow-unchanged", dest="reject_unchanged", action="store_false")
    parser.add_argument("--refine-attempts", type=int, default=1)
    parser.add_argument(
        "--ape-dataset",
        default="",
        help="Path to JSON dataset for prompt optimization",
    )
    parser.add_argument(
        "--batch-dataset",
        default="",
        help="Path to a QuixBugs-style dataset JSON for batch repair generation",
    )
    parser.add_argument(
        "--batch-output-dir",
        default="",
        help="Directory where generated repair candidates and manifest will be written",
    )
    parser.add_argument(
        "--batch-workers",
        type=int,
        default=0,
        help="Number of parallel workers for batch repair generation",
    )
    parser.add_argument("--ape-iterations", type=int, default=2)
    parser.add_argument("--ape-min-score", type=float, default=1.0)
    parser.add_argument("--ape-few-shot-k", type=int, default=2)
    parser.add_argument("--ape-instruction", default="")
    parser.add_argument(
        "--ape-instruction-file",
        default="",
        help="Path to read/write the best APE instruction",
    )
    parser.add_argument("--ape-optimizer-model", default="")
    parser.add_argument(
        "--openrouter-max-tokens",
        type=int,
        default=None,
        help="Limit OpenRouter output tokens to control cost",
    )
    parser.add_argument(
        "--hf-device",
        type=int,
        default=None,
        help="Transformers device id: -1 CPU, 0 GPU, None auto",
    )
    parser.add_argument("--hf-max-new-tokens", type=int, default=512)
    parser.add_argument("--hf-temperature", type=float, default=0.2)
    parser.set_defaults(cache=True, early_stop=True, reject_unchanged=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    instruction_file = (
        Path(args.ape_instruction_file).expanduser()
        if args.ape_instruction_file.strip()
        else None
    )
    cached_instruction = None
    if instruction_file and instruction_file.exists():
        cached_instruction = instruction_file.read_text(encoding="utf-8").strip() or None

    hf_device = args.hf_device
    if hf_device is None:
        hf_device = _detect_transformers_device()

    model_name = _resolve_model_name(args.model, args.model_preset)

    generator = LLMGenerator(
        model_name,
        provider=args.provider,
        enable_cache=args.cache,
        max_patch_chars=args.max_patch_chars,
        max_attempts=args.llm_retries,
        reject_unchanged=args.reject_unchanged,
        transformers_max_new_tokens=args.hf_max_new_tokens,
        transformers_temperature=args.hf_temperature,
        transformers_device=hf_device,
        openrouter_max_tokens=args.openrouter_max_tokens,
    )

    instruction_override = args.ape_instruction.strip() or cached_instruction or DEFAULT_INSTRUCTION
    ape_examples = []

    if args.ape_dataset:
        dataset_path = Path(args.ape_dataset)
        if not dataset_path.exists():
            raise SystemExit(f"APE dataset not found: {dataset_path}")

        ape_examples = DatasetLoader().load_from_json(str(dataset_path))
        try:
            instruction_override = _optimize_instruction_from_dataset(
                generator=generator,
                dataset=ape_examples,
                initial_instruction=instruction_override,
                optimizer_model=args.ape_optimizer_model,
                iterations=args.ape_iterations,
                min_score=args.ape_min_score,
                few_shot_k=args.ape_few_shot_k,
            )
            if instruction_file:
                instruction_file.write_text(instruction_override, encoding="utf-8")
            logging.info("APE best instruction optimized from %d examples", len(ape_examples))
        except Exception as exc:
            logging.warning("APE optimization skipped: %s", exc)

    if args.batch_dataset.strip():
        if not args.batch_output_dir.strip():
            raise SystemExit("--batch-output-dir is required when --batch-dataset is set")

        if args.batch_workers < 0:
            raise SystemExit("--batch-workers must be >= 0")

        batch_runner = BatchRepairRunner(
            generator=generator,
            instruction_override=instruction_override,
            default_package_name=args.package_name.strip() or "python_programs",
            ape_examples=ape_examples,
            ape_few_shot_k=args.ape_few_shot_k,
            mcts_iterations=args.iterations,
            mcts_parallelism=args.parallel,
            stop_on_pass=args.early_stop,
            max_refine_attempts=args.refine_attempts,
            workers=args.batch_workers,
        )
        summary = batch_runner.run_from_json(args.batch_dataset, args.batch_output_dir)
        print("Batch repair completed.")
        print(f"- total: {summary.total}")
        print(f"- output_dir: {summary.output_dir}")
        print(f"- manifest_path: {summary.manifest_path}")
        print(f"- generated_count: {summary.generated_count}")
        print(f"- error_count: {summary.error_count}")
        return

    code_path = Path(args.code)
    if not code_path.exists():
        raise SystemExit(f"Code file not found: {code_path}")
    buggy_code = code_path.read_text(encoding="utf-8")

    tests_path = Path(args.tests)
    if not tests_path.exists():
        raise SystemExit(f"Test file/path not found: {tests_path}")

    module_name = args.module_name.strip() or code_path.stem
    package_name = args.package_name.strip() or None
    if package_name is None and code_path.parent.name == "python_programs":
        package_name = "python_programs"

    root = Node(state=buggy_code)
    tester = Tester(args.tests, module_name=module_name, package_name=package_name)
    db = Database(args.storage)

    bug_description = args.bug_description.strip()
    if not bug_description:
        _, _, initial_output = tester.run_with_details(buggy_code)
        if initial_output:
            bug_description = initial_output
    error_message = bug_description

    total_iterations = 0
    pass_count = 0

    few_shot_examples = ""
    if ape_examples:
        selector = ExampleSelector(max_examples=args.ape_few_shot_k)
        query_example = BugExample(
            buggy_code=buggy_code,
            error_message=error_message,
            correct_code="",
        )
        selected = selector.select_examples(
            query_example, ape_examples, k=args.ape_few_shot_k
        )
        few_shot_examples = selector.format_examples(selected)

    def _generate(code: str) -> str:
        return generator.generate(
            code,
            bug_description=bug_description,
            instruction=instruction_override,
            few_shot_examples=few_shot_examples,
            error_message=error_message,
        )

    def _refine(code: str, feedback: str) -> str:
        return generator.generate(
            code,
            bug_description=bug_description,
            feedback=feedback,
            instruction=instruction_override,
            few_shot_examples=few_shot_examples,
            error_message=error_message,
        )

    def _reward(code: str) -> tuple[int, str, str]:
        return tester.run_with_details(code)

    def _record(record: dict) -> None:
        nonlocal total_iterations, pass_count
        total_iterations += 1
        if record.get("status") == "PASS":
            pass_count += 1
        db.append("iterations", record)
        if record.get("status") == "PASS":
            db.append("solutions", record)
        if record.get("is_best"):
            db.save("best", record)

    start_time = time.perf_counter()
    mcts = MCTS(
        root,
        generator=_generate,
        reward_fn=_reward,
        record_fn=_record,
        refine_fn=_refine,
        parallelism=args.parallel,
        stop_on_pass=args.early_stop,
        max_refine_attempts=args.refine_attempts,
    )
    mcts.run(args.iterations)
    elapsed = time.perf_counter() - start_time

    best_patch = mcts.best_patch or ""
    success_rate = (pass_count / total_iterations) if total_iterations else 0.0

    print("Best patch:")
    print(best_patch if best_patch else "(none)")
    print("Stats:")
    print(f"- iterations: {total_iterations}")
    print(f"- time_sec: {elapsed:.2f}")
    print(f"- success_rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()
