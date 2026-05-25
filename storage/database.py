from __future__ import annotations

import json
import os
from typing import Any, Dict


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_parent()

    def save(self, key: str, value: Any) -> None:
        """Persist a value under a key."""
        data = self._read_all()
        data[key] = value
        self._write_all(data)

    def load(self, key: str) -> Any:
        """Load a value by key."""
        data = self._read_all()
        return data.get(key)

    def append(self, key: str, value: Any) -> None:
        """Append a value to a list under a key."""
        data = self._read_all()
        items = data.get(key)
        if items is None:
            items = []
        elif not isinstance(items, list):
            items = [items]

        items.append(value)
        data[key] = items
        self._write_all(data)

    def _ensure_parent(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _read_all(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}

        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_all(self, data: Dict[str, Any]) -> None:
        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2)
        os.replace(temp_path, self.path)
