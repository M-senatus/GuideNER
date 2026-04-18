"""Central prompt registry for dataset-specific GuideNER prompt templates."""

from __future__ import annotations


REQUIRED_PROMPT_KEYS = ("rule_summary", "rule_validate", "rule_infer")


PROMPTS: dict[str, dict[str, str]] = {
    "conll2003": {
        "rule_summary": """Task: Summarize the generic rules for each named entity category for the named entity recognition task based on the provided text and their corresponding annotations. The output must be structured in JSON format, where the keys represent the entity categories, and the values are lists of rules that have been summarized from the input text and their annotations.

Guidelines: 
(1) Avoid including specific entity names in the output and instead describe general patterns or features. 
(2) Only summarize rules for the entity categories that appear in the provided annotations. Do not include rules for any other categories.
(3) For each annotation provided, generate exactly one summarized rule corresponding to that label.
(4) The order of the summarized rules should strictly correspond to the order of the annotations, and the number of summarized rules must match the number of annotations.

Examples: 
Input Text: EU rejects German call to boycott British lamb . 
Annotations: [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]. 
Output: {{"organization": ["union"], "miscellaneous": ["ethnic groups", "ethnic groups"]}}
Input Text: Iraq 's Saddam meets Russia 's Zhirinovsky .
Annotations: [["Iraq", "location"], ["Saddam", "person"], ["Russia", "location"], ["Zhirinovsky", "person"]]
Output: {{"location": ["country", "country"], "person": ["name", "name"]}}
Input Text: S&P = DENOMS ( K ) 1-10-100 SALE LIMITS US / UK / CA
Annotations: [["S&P", "organization"], ["US", "location"], ["UK", "location"], ["CA", "location"]]
Output: {{"organization": ["financial institution"], "location": ["country", "country", "country"]}}

Summarize for:
Input Text: {input_text}
Annotations: {input_annotations}
Output:
""",
        "rule_validate": """Task: Please identify Person, Organization, Location and Miscellaneous Entity from the given text and rules.
The rules are in JSON format where the key is the entity category and the value is the schema contained in that category.

Examples:
Input Text: EU rejects German call to boycott British lamb.
Rules: {{"organization": ["union"], "miscellaneous": ["nationality"]}}
Output: [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]
Input Text: S&P = DENOMS ( K ) 1-10-100 SALE LIMITS US / UK / CA
Rules: {{"organization": ["financial institution"], "location": ["country", "country", "country"]}}
Output: [["Iraq", "location"], ["Saddam", "person"], ["Russia", "location"], ["Zhirinovsky", "person"]]
Input Text: -- E. Auchard , Wall Street bureau , 212-859-1736
Rules: {{"person": ["journalist"], "organization": ["newspaper bureau"]}}
Output: [["E. Auchard", "person"], ["Wall Street bureau", "organization"]]

Instructions:

Input Text: {input_text}
Rules: {summarized_rules}
Output:
""",
        "rule_infer": """
Task:
Extract named entities from the input text using only the provided rules.

Entity types:
- person
- organization
- location
- miscellaneous

Rules:
{Rules}

Instructions:
1. Extract an entity only if it is explicitly supported by the provided rules.
2. Do not infer new patterns, categories, or entities beyond the rules.
3. Do not use background knowledge or guess.
4. Each extracted entity must be a contiguous span copied exactly from the input text.
5. Return entities in the order they appear in the input text.
6. Use only these lowercase labels: "person", "organization", "location", "miscellaneous".
7. If no entity matches the rules, return an empty list: [].
8. Output only a valid JSON array.
9. Do not explain your reasoning.
10. Do not restate the input text.
11. Do not list tokens or describe rule application.
12. Do not output any text before or after the JSON array.

Examples:

Input Text: EU rejects German call to boycott British lamb.
Output: [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]

Input Text: -- E. Auchard , Wall Street bureau , 212-859-1736
Output: [["E. Auchard", "person"], ["Wall Street bureau", "organization"]]

Input Text: the and of in
Output: []

Now process the following input.

Input Text: {input_text}
Output:
""",
    },
}


def get_prompts(dataset_name: str) -> dict[str, str]:
    """Return the required prompt set for one dataset or fail with a clear error."""
    if dataset_name not in PROMPTS:
        raise KeyError(
            f"Dataset '{dataset_name}' is not registered in prompts.py. "
            f"Available datasets: {sorted(PROMPTS.keys())}"
        )

    dataset_prompts = PROMPTS[dataset_name]
    missing_keys = [
        key
        for key in REQUIRED_PROMPT_KEYS
        if key not in dataset_prompts or not isinstance(dataset_prompts[key], str) or not dataset_prompts[key].strip()
    ]
    if missing_keys:
        raise KeyError(
            f"Dataset '{dataset_name}' is missing required prompt(s) in prompts.py: {missing_keys}"
        )
    return dataset_prompts
