# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingImports=false
from copy import copy

from apps.evaluations.models import Evaluation


def get_task_objectives(evaluation: Evaluation) -> list[str]:
    objectives = getattr(evaluation, "task_objectives", None) or []
    if len(objectives) == 1 and isinstance(objectives[0], list):
        objectives = objectives[0]
    normalized = [
        str(item.get("objective", "")).strip()
        for item in objectives
        if isinstance(item, dict) and str(item.get("objective", "")).strip()
    ]
    return normalized or [evaluation.use_case_description]


def evaluation_for_objective(evaluation: Evaluation, objective: str) -> Evaluation:
    objective_evaluation = copy(evaluation)
    objective_evaluation.use_case_description = objective
    objective_evaluation.task_objectives = []
    return objective_evaluation
