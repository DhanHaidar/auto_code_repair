from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Task:
    task_id: str
    duration: int
    deps: Tuple[str, ...] = field(default_factory=tuple)
    lag_by_dep: Dict[str, int] = field(default_factory=dict)

    def lag_for(self, dep: str) -> int:
        return int(self.lag_by_dep.get(dep, 0))


@dataclass(frozen=True)
class TaskSchedule:
    task_id: str
    start: int
    finish: int


class ProjectPlanner:
    def __init__(self, tasks: Iterable[Task]) -> None:
        self.tasks_by_id: Dict[str, Task] = {}
        for task in tasks:
            if task.task_id in self.tasks_by_id:
                raise ValueError(f"Duplicate task id: {task.task_id}")
            if task.duration < 0:
                raise ValueError(f"Negative duration for task {task.task_id}")
            self.tasks_by_id[task.task_id] = task

        self._build_graph()
        self._validate_dependencies()
        self._validate_no_cycle()

    def _build_graph(self) -> None:
        self.successors: Dict[str, List[str]] = {
            task_id: [] for task_id in self.tasks_by_id
        }
        self.indegree: Dict[str, int] = {task_id: 0 for task_id in self.tasks_by_id}

        for task in self.tasks_by_id.values():
            for dep in task.deps:
                self.successors.setdefault(dep, []).append(task.task_id)
                self.indegree[task.task_id] = self.indegree.get(task.task_id, 0) + 1

    def _validate_dependencies(self) -> None:
        for task in self.tasks_by_id.values():
            for dep in task.deps:
                if dep not in self.tasks_by_id:
                    raise ValueError(f"Unknown dependency '{dep}' for task {task.task_id}")

    def _validate_no_cycle(self) -> None:
        order = self.topological_order()
        if len(order) != len(self.tasks_by_id):
            raise ValueError("Cycle detected in task dependencies.")

    def topological_order(self) -> List[str]:
        indegree = dict(self.indegree)
        ready = sorted([task_id for task_id, deg in indegree.items() if deg == 0])
        order: List[str] = []

        while ready:
            current = ready.pop(0)
            order.append(current)
            for succ in self.successors.get(current, []):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
                    ready.sort()

        return order

    def compute_earliest_times(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        earliest_start: Dict[str, int] = {}
        earliest_finish: Dict[str, int] = {}

        for task_id in self.topological_order():
            task = self.tasks_by_id[task_id]
            if not task.deps:
                earliest_start[task_id] = 0
            else:
                ready_times = [
                    earliest_finish[dep] + task.lag_for(dep) for dep in task.deps
                ]
                earliest_start[task_id] = min(ready_times)
            earliest_finish[task_id] = earliest_start[task_id] + task.duration

        return earliest_start, earliest_finish

    def critical_path_length(self) -> int:
        _, earliest_finish = self.compute_earliest_times()
        return max(earliest_finish.values(), default=0)

    def compute_latest_times(
        self, project_end: Optional[int] = None
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        _, earliest_finish = self.compute_earliest_times()
        if project_end is None:
            project_end = max(earliest_finish.values(), default=0)

        latest_start: Dict[str, int] = {}
        latest_finish: Dict[str, int] = {}

        for task_id in reversed(self.topological_order()):
            task = self.tasks_by_id[task_id]
            succs = self.successors.get(task_id, [])
            if not succs:
                latest_finish[task_id] = project_end
            else:
                candidate_lf: List[int] = []
                for succ_id in succs:
                    succ_task = self.tasks_by_id[succ_id]
                    lag = succ_task.lag_for(task_id)
                    candidate_lf.append(latest_start[succ_id] - lag)
                latest_finish[task_id] = min(candidate_lf)
            latest_start[task_id] = latest_finish[task_id] - task.duration

        return latest_start, latest_finish

    def slack_by_task(self) -> Dict[str, int]:
        earliest_start, _ = self.compute_earliest_times()
        latest_start, _ = self.compute_latest_times()
        return {
            task_id: latest_start[task_id] - earliest_start[task_id]
            for task_id in self.tasks_by_id
        }

    def critical_path_tasks(self) -> List[str]:
        slack = self.slack_by_task()
        return sorted([task_id for task_id, value in slack.items() if value == 0])

    def schedule(self) -> List[TaskSchedule]:
        earliest_start, earliest_finish = self.compute_earliest_times()
        schedules = [
            TaskSchedule(task_id, earliest_start[task_id], earliest_finish[task_id])
            for task_id in self.topological_order()
        ]
        return sorted(schedules, key=lambda item: (item.start, item.task_id))

    def parallel_batches(self) -> List[List[str]]:
        batches: Dict[int, List[str]] = {}
        for item in self.schedule():
            batches.setdefault(item.start, []).append(item.task_id)
        return [sorted(batches[start]) for start in sorted(batches)]
