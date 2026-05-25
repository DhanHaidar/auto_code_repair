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

## Build QuixBugs dataset

```powershell
python tools/build_quixbugs_dataset.py --quixbugs-root QuixBugs --output ape_quixbugs_python_dataset.json
```
