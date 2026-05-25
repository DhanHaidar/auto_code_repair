from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

from core.node import Node

RewardResult = Union[float, Tuple[float, str], Tuple[float, str, str]]


@dataclass
class SimulationResult:
    patch: Optional[str]
    reward: float
    status: str
    output: str


class MCTS:
    def __init__(
        self,
        root: Node,
        generator: Optional[Callable[[str], str]] = None,
        reward_fn: Optional[Callable[[str], RewardResult]] = None,
        record_fn: Optional[Callable[[Dict[str, object]], None]] = None,
        refine_fn: Optional[Callable[[str, str], str]] = None,
        parallelism: int = 1,
        stop_on_pass: bool = True,
        max_refine_attempts: int = 1,
    ) -> None:
        self.root = root
        self.generator = generator
        self.reward_fn = reward_fn
        self.exploration_constant = 0.7
        self.solutions: List[str] = []
        self.record_fn = record_fn
        self.refine_fn = refine_fn
        self.logger = logging.getLogger(__name__)
        self.best_reward = float("-inf")
        self.best_patch: Optional[str] = None
        self.parallelism = max(1, int(parallelism))
        self.stop_on_pass = stop_on_pass
        self.max_refine_attempts = max(0, int(max_refine_attempts))

    def search(self, iterations: int) -> None:
        """Run MCTS search for a number of iterations."""
        self.run(iterations)

    def run(self, iterations: int) -> None:
        """Run the main MCTS loop for a number of iterations."""
        if self.parallelism <= 1:
            for iteration in range(1, iterations + 1):
                leaf = self.select(self.root)
                child = self.expand(leaf)
                result = self.simulate(child)
                self.backpropagate(child, result.reward)
                if self._handle_result(iteration, result):
                    break
            return

        iteration = 0
        with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
            while iteration < iterations:
                batch_size = min(self.parallelism, iterations - iteration)
                batch: List[Tuple[Node, object]] = []
                for _ in range(batch_size):
                    leaf = self.select(self.root)
                    child = self.expand(leaf)
                    future = executor.submit(self.simulate, child)
                    batch.append((child, future))

                stop = False
                for child, future in batch:
                    result = future.result()
                    iteration += 1
                    self.backpropagate(child, result.reward)
                    if self._handle_result(iteration, result):
                        stop = True
                if stop:
                    break

    def select(self, node: Node) -> Node:
        """Select a node to expand based on a policy."""
        current = node
        while not current.is_leaf():
            current = self._best_uct_child(current)
        return current

    def expand(self, node: Node) -> Node:
        """Expand the given node by adding a child."""
        child = Node(state=node.state)
        node.add_child(child)
        return child

    def simulate(self, node: Node) -> SimulationResult:
        """Simulate a rollout from the node and return a result."""
        if self.generator is None:
            return SimulationResult(patch=None, reward=0.0, status="SKIP", output="")

        patch = self.generator(str(node.state))
        node.state = patch
        base_result = self._evaluate_patch(patch)

        if (
            self.refine_fn is not None
            and self.max_refine_attempts > 0
            and base_result.output
            and base_result.reward < 1.0
        ):
            best_patch = patch
            best_result = base_result
            for _ in range(self.max_refine_attempts):
                refined = self.refine_fn(best_patch, best_result.output)
                refined_result = self._evaluate_patch(refined)
                if refined_result.reward > best_result.reward:
                    best_result = refined_result
                    best_patch = refined
                if refined_result.reward >= 1.0:
                    best_result = refined_result
                    best_patch = refined
                    break

            node.state = best_patch
            return SimulationResult(
                patch=best_patch,
                reward=best_result.reward,
                status=best_result.status,
                output=best_result.output,
            )

        return base_result

    def _evaluate_patch(self, patch: str) -> SimulationResult:
        if self.reward_fn is None:
            return SimulationResult(patch=patch, reward=0.0, status="UNKNOWN", output="")

        result = self.reward_fn(patch)
        output = ""
        if isinstance(result, tuple):
            reward = float(result[0]) if len(result) > 0 else 0.0
            status = result[1] if len(result) > 1 else "UNKNOWN"
            if len(result) > 2 and result[2] is not None:
                output = str(result[2])
        else:
            reward, status = float(result), "UNKNOWN"

        return SimulationResult(
            patch=patch, reward=float(reward), status=status, output=output
        )

    def backpropagate(self, node: Node, reward: float) -> None:
        """Backpropagate the simulation reward up the tree."""
        current: Optional[Node] = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def _best_uct_child(self, node: Node) -> Node:
        parent_visits = max(1, node.visits)
        best_score = float("-inf")
        best_child = node.children[0]

        for child in node.children:
            if child.visits == 0:
                return child

            exploitation = child.value / child.visits
            exploration = self.exploration_constant * math.sqrt(
                math.log(parent_visits) / child.visits
            )
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _handle_result(self, iteration: int, result: SimulationResult) -> bool:
        patch = result.patch
        is_best = False

        if patch is not None and result.reward >= 1.0:
            self.solutions.append(patch)

        if result.reward > self.best_reward:
            self.best_reward = result.reward
            self.best_patch = patch
            is_best = True
            if patch is not None:
                self._log_best_patch(result.reward, patch, result.status)

        self._log_iteration(iteration, result, is_best)

        if self.record_fn is not None and patch is not None:
            self.record_fn(
                {
                    "iteration": iteration,
                    "patch": patch,
                    "reward": result.reward,
                    "status": result.status,
                    "output": result.output,
                    "is_best": is_best,
                }
            )

        if self.stop_on_pass and result.reward >= 1.0:
            self.logger.info("Early stopping: PASS patch found.")
            return True

        return False

    def _log_iteration(
        self, iteration: int, result: SimulationResult, is_best: bool
    ) -> None:
        patch_len = len(result.patch) if result.patch is not None else 0
        output_len = len(result.output) if result.output else 0
        self.logger.info(
            "Iteration %d | reward=%s | status=%s | patch_len=%d | output_len=%d | best=%s",
            iteration,
            result.reward,
            result.status,
            patch_len,
            output_len,
            is_best,
        )

    def _log_best_patch(self, reward: float, patch: str, status: str) -> None:
        first_line = patch.strip().splitlines()[0] if patch.strip() else ""
        if len(first_line) > 120:
            first_line = f"{first_line[:120]}..."
        self.logger.info(
            "New best patch | reward=%s | status=%s | first_line=%s",
            reward,
            status,
            first_line,
        )
