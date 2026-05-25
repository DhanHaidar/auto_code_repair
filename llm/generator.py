from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

from llm.prompt import build_prompt

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None


class LLMGenerator:
    def __init__(
        self,
        model_name: str,
        provider: str = "auto",
        enable_cache: bool = True,
        max_patch_chars: int = 8000,
        max_attempts: int = 2,
        reject_unchanged: bool = True,
        transformers_max_new_tokens: int = 512,
        transformers_temperature: float = 0.2,
        transformers_device: int = -1,
        openrouter_max_tokens: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.provider = provider.lower()
        self.enable_cache = enable_cache
        self.max_patch_chars = max_patch_chars
        self.max_attempts = max(1, int(max_attempts))
        self.reject_unchanged = reject_unchanged
        self.transformers_max_new_tokens = transformers_max_new_tokens
        self.transformers_temperature = transformers_temperature
        self.transformers_device = transformers_device
        self.openrouter_max_tokens = openrouter_max_tokens
        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._hf_pipeline: Optional[Any] = None
        self._hf_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._openrouter_validated = False

    def generate(
        self,
        buggy_code: str,
        bug_description: str = "",
        feedback: str = "",
        instruction: Optional[str] = None,
        few_shot_examples: str = "",
        error_message: str = "",
        prompt_override: Optional[str] = None,
    ) -> str:
        """Generate a candidate patch from buggy code using an LLM."""
        if self.provider == "openrouter" and not self._get_openrouter_api_key():
            raise RuntimeError(
                "OpenRouter API key is not set. Set OPENROUTER_API_KEY or "
                "create secrets/openrouter.key."
            )
        if self.provider == "openai" and not self._get_openai_api_key():
            raise RuntimeError(
                "OpenAI API key is not set. Set OPENAI_API_KEY or "
                "create secrets/openai.key."
            )
        if self.provider == "openrouter":
            self._validate_openrouter_model()
        cache_key = self._make_cache_key(
            buggy_code,
            bug_description,
            feedback,
            instruction,
            few_shot_examples,
            error_message,
            prompt_override,
        )
        expected_symbols = self._extract_expected_symbols(buggy_code)
        if self.enable_cache:
            cached = self._get_cached(cache_key)
            if cached is not None and self._is_patch_valid(
                cached, expected_symbols, buggy_code
            ):
                return cached

        if prompt_override is not None:
            prompt = prompt_override
        else:
            prompt = build_prompt(
                bug_description,
                buggy_code,
                feedback,
                instruction=instruction,
                few_shot_examples=few_shot_examples,
                error_message=error_message,
            )
        last_patch: Optional[str] = None

        for attempt in range(self.max_attempts):
            attempt_prompt = prompt
            if attempt > 0:
                attempt_prompt = self._tighten_prompt(prompt, expected_symbols)

            patch = None
            if self.provider in {"auto", "openai"}:
                patch = self._generate_openai(attempt_prompt, buggy_code)
            if patch is None and self.provider in {"auto", "openrouter"}:
                patch = self._generate_openrouter(attempt_prompt, buggy_code)
            if patch is None and self.provider in {"auto", "transformers"}:
                patch = self._generate_transformers(attempt_prompt, buggy_code)

            if patch is None:
                continue

            patch = self._limit_patch(patch)
            patch = self._apply_guided_replacements(patch, bug_description)
            last_patch = patch

            if self._is_patch_valid(patch, expected_symbols, buggy_code):
                if self.enable_cache:
                    self._set_cached(cache_key, patch)
                return patch

        fallback = last_patch or self._dummy_patch(buggy_code)
        fallback = self._limit_patch(fallback)
        if self.enable_cache:
            self._set_cached(cache_key, fallback)
        return fallback

    def _generate_openai(self, prompt: str, fallback_code: str) -> Optional[str]:
        api_key = self._get_openai_api_key()
        if OpenAI is None or not api_key:
            return None

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an automated program repair assistant. "
                        "Think step-by-step but do not reveal chain-of-thought. "
                        "Return results in the requested format."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content or ""
        return self._extract_patch(content, fallback_code)

    def _generate_openrouter(self, prompt: str, fallback_code: str) -> Optional[str]:
        api_key = self._get_openrouter_api_key()
        if OpenAI is None or not api_key:
            return None

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        headers = {}
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers or None,
        )
        request: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an automated program repair assistant. "
                        "Think step-by-step but do not reveal chain-of-thought. "
                        "Return results in the requested format."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        if self.openrouter_max_tokens is not None:
            request["max_tokens"] = int(self.openrouter_max_tokens)
        response = client.chat.completions.create(**request)

        content = response.choices[0].message.content or ""
        return self._extract_patch(content, fallback_code)

    def _validate_openrouter_model(self) -> None:
        if self._openrouter_validated:
            return

        if os.getenv("OPENROUTER_VALIDATE_MODEL", "true").lower() in {
            "0",
            "false",
            "no",
        }:
            self._openrouter_validated = True
            return

        api_key = self._get_openrouter_api_key()
        if OpenAI is None or not api_key:
            return

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        headers = {}
        site_url = os.getenv("OPENROUTER_SITE_URL")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers or None,
        )

        try:
            models = client.models.list()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to validate OpenRouter model '{self.model_name}': {exc}"
            )

        model_ids = set()
        data = getattr(models, "data", None) or []
        for item in data:
            model_id = getattr(item, "id", None)
            if model_id:
                model_ids.add(model_id)

        if self.model_name not in model_ids:
            raise RuntimeError(
                "OpenRouter model not found: "
                f"{self.model_name}. Check your model ID."
            )

        self._openrouter_validated = True

    def _get_openrouter_api_key(self) -> Optional[str]:
        return self._get_api_key(
            "OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY_FILE",
            os.path.join("secrets", "openrouter.key"),
        )

    def _get_openai_api_key(self) -> Optional[str]:
        return self._get_api_key(
            "OPENAI_API_KEY",
            "OPENAI_API_KEY_FILE",
            os.path.join("secrets", "openai.key"),
        )

    def _get_api_key(
        self, env_name: str, file_env_name: str, default_path: str
    ) -> Optional[str]:
        api_key = os.getenv(env_name)
        if api_key:
            return api_key

        key_path = os.getenv(file_env_name) or default_path
        return self._read_key_file(key_path)

    def _read_key_file(self, path: str) -> Optional[str]:
        if not path:
            return None

        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            expanded = os.path.abspath(expanded)

        try:
            with open(expanded, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
                return content or None
        except OSError:
            return None

    def _generate_transformers(self, prompt: str, fallback_code: str) -> Optional[str]:
        generator = self._get_transformers_pipeline()
        if generator is None:
            return None

        try:
            outputs = generator(
                prompt,
                max_new_tokens=self.transformers_max_new_tokens,
                do_sample=True,
                temperature=self.transformers_temperature,
                return_full_text=False,
            )
        except TypeError:
            outputs = generator(
                prompt,
                max_new_tokens=self.transformers_max_new_tokens,
                do_sample=True,
                temperature=self.transformers_temperature,
            )
        text = outputs[0].get("generated_text", "")

        completion = text[len(prompt) :] if text.startswith(prompt) else text
        return self._extract_patch(completion, fallback_code)

    def _extract_patch(self, text: str, fallback_code: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return fallback_code

        fenced = self._first_fenced_block(cleaned)
        if fenced is not None:
            return self._normalize_patch(fenced, fallback_code)

        match = re.search(r"PATCH:\s*(.+)", cleaned, re.IGNORECASE | re.DOTALL)
        if match:
            return self._normalize_patch(match.group(1), fallback_code)

        code_section = self._strip_to_code(cleaned)
        return self._normalize_patch(code_section, fallback_code)

    def _dummy_patch(self, buggy_code: str) -> str:
        return buggy_code

    def _extract_expected_symbols(self, code: str) -> List[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        symbols: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(node.name)
        return symbols

    def _is_patch_valid(
        self, patch: str, expected_symbols: List[str], buggy_code: str
    ) -> bool:
        if self.reject_unchanged and patch.strip() == buggy_code.strip():
            return False

        try:
            tree = ast.parse(patch)
        except SyntaxError:
            return False

        if expected_symbols:
            defined = {
                node.name
                for node in tree.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            }
            if not all(name in defined for name in expected_symbols):
                return False

        return True

    def _tighten_prompt(self, prompt: str, expected_symbols: List[str]) -> str:
        symbols = ", ".join(expected_symbols) if expected_symbols else ""
        strict = (
            "\n\nSTRICT OUTPUT RULES:\n"
            "- Output ONLY valid Python code.\n"
            "- Do NOT include markdown fences or commentary.\n"
            "- Do NOT include example outputs or print statements.\n"
            "- Do NOT repeat the code.\n"
        )
        if symbols:
            strict += f"- Must define these symbols: {symbols}.\n"
        return prompt + strict

    def _apply_guided_replacements(self, patch: str, bug_description: str) -> str:
        if not bug_description:
            return patch

        instructions = bug_description.strip()
        replacements = self._extract_replacements(instructions)
        if not replacements:
            return patch

        updated = patch
        for old, new in replacements:
            if old and new and old in updated:
                updated = updated.replace(old, new)

        return updated

    def _extract_replacements(self, text: str) -> List[tuple[str, str]]:
        pattern = re.compile(
            r"(?:ubah|ganti)\s+['\"]?(.*?)['\"]?\s+(?:menjadi|dengan)\s+['\"]?(.*?)['\"]?(?:\.|$)",
            re.IGNORECASE,
        )
        results = []
        for match in pattern.finditer(text):
            old = match.group(1).strip()
            new = match.group(2).strip()
            if old and new:
                results.append((old, new))
        return results

    def _first_fenced_block(self, text: str) -> Optional[str]:
        match = re.search(
            r"```(?:python)?\s*\n([\s\S]*?)```",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)

        match = re.search(
            r"```(?:python)?\s*([\s\S]*?)```",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)

        return None

    def _strip_to_code(self, text: str) -> str:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(
                (
                    "def ",
                    "class ",
                    "from ",
                    "import ",
                    "@",
                    "if __name__",
                    '"""',
                    "'''",
                )
            ):
                return "\n".join(lines[index:])
        return text

    def _normalize_patch(self, patch: str, fallback_code: str) -> str:
        cleaned = patch.strip()
        if not cleaned:
            return fallback_code

        cleaned = self._remove_fence_lines(cleaned)
        cleaned = self._dedupe_repeated_block(cleaned)
        return cleaned.strip() if cleaned.strip() else fallback_code

    def _remove_fence_lines(self, text: str) -> str:
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines)

    def _dedupe_repeated_block(self, text: str) -> str:
        lines = text.splitlines()
        first_line = ""
        for line in lines:
            if line.strip():
                first_line = line
                break
        if not first_line:
            return text

        joined = "\n".join(lines)
        marker = f"\n{first_line}"
        idx = joined.find(marker, 1)
        if idx != -1 and idx > 40:
            return joined[:idx].rstrip()
        return text

    def _get_transformers_pipeline(self) -> Optional[Any]:
        if pipeline is None:
            return None

        if self._hf_pipeline is not None:
            return self._hf_pipeline

        with self._hf_lock:
            if self._hf_pipeline is not None:
                return self._hf_pipeline

            kwargs = {"model": self.model_name}
            if self.transformers_device is not None:
                kwargs["device"] = self.transformers_device
            self._hf_pipeline = pipeline("text-generation", **kwargs)
            return self._hf_pipeline

    def _make_cache_key(
        self,
        buggy_code: str,
        bug_description: str,
        feedback: str,
        instruction: Optional[str],
        few_shot_examples: str,
        error_message: str,
        prompt_override: Optional[str],
    ) -> str:
        payload = "|".join(
            [
                self.model_name,
                self.provider,
                str(self.max_patch_chars),
                bug_description,
                feedback,
                instruction or "",
                few_shot_examples,
                error_message,
                prompt_override or "",
                buggy_code,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached(self, key: str) -> Optional[str]:
        with self._cache_lock:
            return self._cache.get(key)

    def _set_cached(self, key: str, value: str) -> None:
        with self._cache_lock:
            self._cache[key] = value

    def _finalize_patch(self, cache_key: str, patch: str) -> str:
        patch = self._limit_patch(patch)
        if self.enable_cache:
            self._set_cached(cache_key, patch)
        return patch

    def _limit_patch(self, patch: str) -> str:
        if self.max_patch_chars and len(patch) > self.max_patch_chars:
            self.logger.warning(
                "Patch length %d exceeds limit %d; truncating.",
                len(patch),
                self.max_patch_chars,
            )
            return patch[: self.max_patch_chars]
        return patch
