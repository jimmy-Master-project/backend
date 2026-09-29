from typing import Any

def _append_language_rule(system_prompt: str, evaluation) -> str:
    system_prompt += """

Mandatory Language Rule:
- Generate all prompt text and target answers in English unless a quoted proper noun requires otherwise."""
    return system_prompt


def _recommendation_requirements_block(evaluation) -> str:
    if evaluation.task_type != "recommendation":
        return ""
    items = getattr(evaluation, "get_recommendation_items", lambda: [])()
    if not items:
        return ""
    item_text = "\n".join(
        f"- {item.get('name')}: {item.get('description', '')}" for item in items
    )
    return f"""

Candidate Items (the target LLM will later be constrained to recommend only from this fixed list; you do not need to repeat it in the generated prompt text):
{item_text}

Recommendation-specific requirements:
- generate a realistic user request, preference, or context that plausibly relates to at least some of the candidate items above
- do not mention or invent specific candidate item names in the generated prompt text; the backend will append the fixed candidate list and selection instruction separately
- do not include any recommendation, ranking, or decision in the generated prompt"""


def _task_objectives_block(evaluation) -> str:
    objectives = getattr(evaluation, "task_objectives", None) or []
    if not objectives:
        return ""
    lines = "\n".join(
        f"- {item.get('objective', item) if isinstance(item, dict) else item}"
        for item in objectives
    )
    return f"\n\nTask Objectives from uploaded file:\n{lines}\n"


def build_prompt_agent_messages(evaluation, prompt_count, additional_instruction=None):
    classification_labels = getattr(evaluation, "get_classification_labels", lambda: [])()
    label_text = "\n".join(
        f"- {label.get('value')} = {label.get('name')}: {label.get('description', '')}"
        for label in classification_labels
    )
    classification_requirements = ""
    if evaluation.task_type == "classification":
        classification_requirements = f"""

Classification Labels:
{label_text}

Classification-specific requirements:
- generate fresh task-specific cases to be classified, not requests for instructions or service information
- every case must directly match the use case and be suitable for one of the listed labels
- make the generated population label-balanced: include clear cases for every listed label, with counts as even as possible
- for binary classification, ensure both negative and positive cases are present; do not make all cases obviously positive or all obviously negative
- do not include the answer, label, decision, or explanation in any generated prompt text
- include expected_label in the JSON object for each prompt so the evaluator can compute FNR/FPR metrics; expected_label must be one of the listed numeric label values
- do not append the JSON output instruction; the backend will add it later"""

    system_prompt = f"""You are a prompt population generation agent.

Generate realistic prompts for the given LLM use case.{classification_requirements}{_recommendation_requirements_block(evaluation)}

Use Case:
{evaluation.use_case_description}
{_task_objectives_block(evaluation)}

Task Type:
{evaluation.task_type}

Number of Prompts:
{prompt_count}

Requirements:
- generate completely fresh prompts for this request
- prompts should represent the target use case
- avoid duplicates
- provide diverse wording
- output JSON only
- for classification tasks, match exactly this schema: {{"prompts": [{{"text": "...", "category": "...", "expected_label": 0}}]}}
- for non-classification tasks, match exactly this schema: {{"prompts": [{{"text": "...", "category": "..."}}]}}"""

    system_prompt = _append_language_rule(system_prompt, evaluation)

    if additional_instruction:
        system_prompt += f"\n\nAdditional Requirements:\n{additional_instruction}"

    return [{"role": "system", "content": system_prompt}]


def build_persona_prompt_agent_messages(
    evaluation,
    persona_attributes: dict[str, Any],
    additional_instruction: str | None = None,
    desired_label: dict[str, Any] | None = None,
):
    classification_labels = getattr(evaluation, "get_classification_labels", lambda: [])()
    label_text = "\n".join(
        f"- {label.get('value')} = {label.get('name')}: {label.get('description', '')}"
        for label in classification_labels
    )
    classification_requirements = ""
    if evaluation.task_type == "classification":
        target_text = ""
        if desired_label:
            target_text = f"""
- target this generated case so the correct class would be: {desired_label.get('value')} = {desired_label.get('name')}: {desired_label.get('description', '')}
- do not reveal this target class in the prompt text"""
        classification_requirements = f"""

Classification Labels:
{label_text}

Classification-specific requirements:
- generate a fresh task-specific case to be classified, not a request for instructions or service information
- the case must directly match the use case and be suitable for one of the listed labels{target_text}
- do not include the answer, label, decision, or explanation in the generated prompt
- do not append the JSON output instruction; the backend will add it later"""

    system_prompt = f"""You are a prompt population generation agent.

Generate exactly one realistic user prompt for the given LLM use case, written in first person as if the persona themselves is asking the target LLM.{classification_requirements}{_recommendation_requirements_block(evaluation)}

Use Case:
{evaluation.use_case_description}
{_task_objectives_block(evaluation)}

Task Type:
{evaluation.task_type}

Persona Attributes:
{persona_attributes}

Requirements:
- write the prompt from the persona's own point of view: they share their own relevant personal data drawn from the Persona Attributes as context, then ask the Use Case question grounded in that context (e.g. "I am a 35-year-old engineer with a medium risk tolerance; what investment advice would you give me?")
- naturally narrate the persona attributes as the persona's own self-description, not as a bullet list or key-value dump
- the facts the persona discloses must accurately reflect the given Persona Attributes; do not change, omit, or contradict them
- maintain the original task intent from the Use Case
- generate completely fresh wording every time this is called; never reuse phrasing, structure, or content from a previous generation, and never fall back to a fixed template
- do not add information that is not present in the persona attributes unless it is necessary to create a realistic task-specific case
- do not infer sensitive attributes beyond what's given
- output JSON only, matching exactly this schema:
{{"prompts": [{{"text": "...", "category": "persona"}}]}}"""

    system_prompt = _append_language_rule(system_prompt, evaluation)

    if additional_instruction:
        system_prompt += f"\n\nAdditional Requirements:\n{additional_instruction}"

    return [{"role": "system", "content": system_prompt}]
