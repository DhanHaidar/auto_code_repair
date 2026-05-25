from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Node:
    state: Any
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return len(self.children) == 0

    def add_child(self, child: "Node") -> "Node":
        """Attach an existing child node to this node."""
        child.parent = self
        self.children.append(child)
        return child

    def update(self, reward: float) -> None:
        """Update visit count and accumulated reward."""
        self.visits += 1
        self.value += reward
