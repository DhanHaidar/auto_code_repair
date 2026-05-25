from __future__ import annotations

from typing import Optional


DEFAULT_INSTRUCTION = (
    "- Jelaskan bug.\n"
    "- Perbaiki bug.\n"
    "- Output berupa kode Python yang sudah diperbaiki, disertai penjelasan\n"
    "  gaya QuixBugs dalam blok docstring (triple quotes) setelah kode.\n"
    "  Contoh struktur docstring: Title, Input, Precondition, Output."
)

APE_TEMPLATE = (
    "Context: kode buggy\n"
    "```python\n"
    "{buggy_code}\n"
    "```\n\n"
    "Instruction:\n"
    "{instruction}\n\n"
    "Few-shot examples:\n"
    "{few_shot_examples}\n\n"
    "Error message:\n"
    "{error_message}\n\n"
    "Konteks tambahan (jika ada):\n"
    "{bug_description}\n\n"
    "{feedback_section}"
    "Output wajib hanya berupa kode Python yang sudah diperbaiki\n"
    "beserta docstring penjelasan (tanpa markdown, tanpa blok ```).\n"
    "Jangan ulangi kode. Jangan tambahkan teks lain di luar kode dan docstring."
)


def build_prompt(
    bug_description: str,
    code_snippet: str,
    feedback: str = "",
    instruction: Optional[str] = None,
    few_shot_examples: str = "",
    error_message: str = "",
) -> str:
    """Build a prompt for the LLM from bug context and code."""
    if instruction is None and not few_shot_examples and not error_message:
        return _build_legacy_prompt(bug_description, code_snippet, feedback)

    description = bug_description.strip()
    description_block = description if description else "(not provided)"
    feedback_block = feedback.strip()
    feedback_section = ""
    if feedback_block:
        feedback_section = f"Kegagalan test:\n{feedback_block}\n\n"

    instruction_block = instruction.strip() if instruction else DEFAULT_INSTRUCTION
    few_shot_block = few_shot_examples.strip() or "(none)"
    error_block = error_message.strip() or "(none)"

    return APE_TEMPLATE.format(
        instruction=instruction_block,
        few_shot_examples=few_shot_block,
        buggy_code=code_snippet,
        error_message=error_block,
        bug_description=description_block,
        feedback_section=feedback_section,
    )


def _build_legacy_prompt(bug_description: str, code_snippet: str, feedback: str) -> str:
    description = bug_description.strip()
    description_block = description if description else "(not provided)"
    feedback_block = feedback.strip()
    feedback_section = ""
    if feedback_block:
        feedback_section = f"Kegagalan test:\n{feedback_block}\n\n"

    return (
        "Context: kode buggy\n"
        "```python\n"
        f"{code_snippet}\n"
        "```\n\n"
        "Instruction:\n"
        f"{DEFAULT_INSTRUCTION}\n\n"
        "Teknik:\n"
        "- Chain of Thought (gunakan secara internal, jangan ditampilkan).\n"
        "- Self-reflection (periksa kembali hasil sebelum menjawab).\n\n"
        "Konteks tambahan (jika ada):\n"
        f"{description_block}\n\n"
        f"{feedback_section}"
        "Output wajib hanya berupa kode Python yang sudah diperbaiki\n"
        "beserta docstring penjelasan (tanpa markdown, tanpa blok ```).\n"
        "Jangan ulangi kode. Jangan tambahkan teks lain di luar kode dan docstring."
    )
