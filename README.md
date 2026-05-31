# APE Prompt Optimizer Demo

This repo now includes a small, runnable Automatic Prompt Optimization (APE)
prototype for APR prompt tuning.

## Quick run

```powershell
python ape_prompt_optimizer.py
```

## Notes

- The demo uses a mock LLM coder by default.
- To use OpenAI for instruction optimization, set `OPENAI_API_KEY` and
  update the client in `run_demo()`.

## APR integration

Run the main APR pipeline with APE enabled:

```powershell
python main.py --code D:\phyton\apr_code_review\QuixBugs\python_programs\pascal.py --test D:\phyton\apr_code_review\QuixBugs\python_testcases\test_pascal.py --provider openrouter --model deepseek/deepseek-v4-flash --ape-dataset D:\phyton\apr_code_review\your_training_dataset.json --ape-iterations 2 --ape-few-shot-k 2
```

Optional flags:

- `--ape-instruction` to override the initial instruction.
- `--ape-optimizer-model` to use OpenAI for instruction mutations.

## Batch QuixBugs workflow

This repo now supports a two-step batch flow for QuixBugs-style datasets:

1. Generate repairs for all bugs in one run and store the candidates in a folder.
2. Score every generated candidate with the plausible-fix and correct-fix evaluator.

Example batch repair run:

```powershell
python main.py --batch-dataset ape_quixbugs_python_dataset.json --batch-output-dir repaired_candidates --provider transformers --model-preset light
```

Example scoring run:

```powershell
python tools/score_quixbugs_repairs.py --dataset ape_quixbugs_python_dataset.json --candidate-dir repaired_candidates --package-name python_programs --workers 8
```

The batch repair run writes one patched `.py` file per bug plus a `batch_results.json` manifest into the output directory. The scoring step reads that folder and reports plausible-fix and correct-fix counts automatically.

## Build QuixBugs dataset

```powershell
python tools/build_quixbugs_dataset.py --quixbugs-root QuixBugs --output ape_quixbugs_python_dataset.json
```
