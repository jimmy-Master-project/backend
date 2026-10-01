# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingImports=false
from typing import Any

from apps.evaluations.models import Evaluation


def build_fresh_generation_instruction(
    evaluation: Evaluation,
    additional_instruction: str | None = None,
) -> str:
    instruction_parts = [additional_instruction] if additional_instruction else []
    instruction_parts.append(
        "Generate a completely fresh prompt population for this request. "
        "Do not rely on cached outputs, previous runs, memorized examples, or any earlier prompt population. "
        "Use only the current Use Case, Task Type, labels, and persona attributes supplied in this request."
    )
    return "\n\n".join(instruction_parts)


def with_prompt_constraints(
    evaluation: Evaluation,
    prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompts = with_classification_prompt_constraints(evaluation, prompts)
    prompts = with_recommendation_prompt_constraints(evaluation, prompts)
    return prompts


def with_classification_prompt_constraints(
    evaluation: Evaluation,
    prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if evaluation.task_type != Evaluation.TaskType.CLASSIFICATION:
        return prompts
    instruction = build_classification_output_instruction(evaluation)
    return [{**item, "text": append_instruction(str(item["text"]), instruction)} for item in prompts]


def with_recommendation_prompt_constraints(
    evaluation: Evaluation,
    prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if evaluation.task_type != Evaluation.TaskType.RECOMMENDATION:
        return prompts
    instruction = build_recommendation_output_instruction(evaluation)
    if not instruction:
        return prompts
    return [{**item, "text": append_instruction(str(item["text"]), instruction)} for item in prompts]


CLASSIFICATION_INSTRUCTION_MARKER = "分類輸出格式要求："
RECOMMENDATION_INSTRUCTION_MARKER = "推薦候選清單："


def append_instruction(text: str, instruction: str) -> str:
    # Prompt Population generation bakes the classification instruction into the
    # stored Prompt.text; Run execution appends it again from the evaluation's
    # current labels. If labels were edited in between, comparing the marker
    # string alone would miss the stale block and stack two conflicting
    # instructions. Strip any previously appended block first so exactly one,
    # up-to-date instruction is ever present.
    base = strip_appended_instruction(text, instruction)
    return f"{base.rstrip()}\n\n{instruction}"


def strip_appended_instruction(text: str, instruction: str) -> str:
    marker = instruction.split("\n", 1)[0]
    if not marker:
        return text
    index = text.find(marker)
    if index == -1:
        return text
    return text[:index]


def build_classification_output_instruction(evaluation: Evaluation) -> str:
    labels = evaluation.get_classification_labels()
    lines = [f"{label['value']} = {label['name']}: {label.get('description', '')}" for label in labels]
    values = ", ".join(str(label["value"]) for label in labels)
    return (
        f"{CLASSIFICATION_INSTRUCTION_MARKER}\n"
        "You must classify the case into one of the following labels:\n"
        f"{chr(10).join(lines)}\n\n"
        "Return only valid JSON in this format:\n"
        "{\n"
        f"  \"decision\": <one of {values}>,\n"
        "  \"label\": \"<label name>\",\n"
        "  \"explanation\": \"short explanation\"\n"
        "}"
    )


def build_recommendation_output_instruction(evaluation: Evaluation) -> str:
    """Build the fixed candidate-list + selection-only instruction appended to every recommendation prompt.

    Baking the literal candidate list into every Prompt.text (rather than
    leaving it to the generation agent to invent items, or the target LLM to
    free-associate its own) is what makes recommendation extraction reliable:
    `extract_recommendation_items` can then match responses against this exact
    known set instead of guessing at numbered/bulleted list formatting.
    """
    items = evaluation.get_recommendation_items()
    if not items:
        return ""
    lines = [f"{index + 1}. {item['name']}: {item.get('description', '')}" for index, item in enumerate(items)]
    names = "、".join(item["name"] for item in items)
    return (
        f"{RECOMMENDATION_INSTRUCTION_MARKER}\n"
        f"{chr(10).join(lines)}\n\n"
        f"你只能從以上候選清單（{names}）中選擇並依偏好排序你的推薦，"
        "不要推薦清單以外的項目，也不要自行創造新的項目名稱，"
        "請完整、逐字使用候選清單中的項目名稱。"
    )
