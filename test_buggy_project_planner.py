import pytest

from buggy_project_planner import ProjectPlanner, Task


def build_sample_plan() -> ProjectPlanner:
    tasks = [
        Task("A", 3),
        Task("B", 4, deps=("A",)),
        Task("C", 2, deps=("A",)),
        Task("D", 5, deps=("B", "C"), lag_by_dep={"C": 1}),
        Task("E", 1, deps=("D",)),
    ]
    return ProjectPlanner(tasks)


def test_critical_path_length_uses_all_dependencies() -> None:
    planner = build_sample_plan()
    assert planner.critical_path_length() == 13


def test_schedule_respects_predecessors_and_lags() -> None:
    planner = build_sample_plan()
    schedule = {item.task_id: item for item in planner.schedule()}
    assert schedule["D"].start == 7
    assert schedule["E"].start == 12


def test_critical_path_tasks() -> None:
    planner = build_sample_plan()
    assert planner.critical_path_tasks() == ["A", "B", "D", "E"]


def test_cycle_detection_raises() -> None:
    tasks = [Task("A", 1, deps=("B",)), Task("B", 1, deps=("A",))]
    with pytest.raises(ValueError):
        ProjectPlanner(tasks)
