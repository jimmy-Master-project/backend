# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingImports=false
from typing import Any


def build_prompt_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source": "prompt_generation_agent"}
    if "expected_label" in item:
        metadata["expected_label"] = item.get("expected_label")
    if "persona_id" in item:
        metadata.update(
            {
                "persona_id": item["persona_id"],
                "persona_attributes": item.get("persona_attributes", {}),
                "missing_attributes": item.get("missing_attributes", []),
                "fairness": item.get("fairness", {}),
            }
        )
    if "objective" in item:
        metadata["task_objective"] = item["objective"]
    return metadata
