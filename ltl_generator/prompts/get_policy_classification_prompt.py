# prompts/get_policy_classification_prompt.py
import json
from typing import Dict

CLASSIFICATION_SYSTEM = """
You are a specialized AI component within a larger application, acting as a **Policy Classification Service**.
Your function is to analyze a single, refined safety policy and classify it against two different frameworks: high-level EU AI Act Ethical Principles and low-level technical Risk Categories.
"""

CLASSIFICATION_USER = """
The application has provided a refined safety policy. Your task is to perform a two-part classification:
1.  Select **all relevant** EU AI Act Ethical Principles.
2.  Select the **single best and most specific** Risk Category that describes the policy's technical intent.

--- EU AI ACT ETHICAL PRINCIPLES (Source: COMPL-AI Framework) ---
-   **Human Agency and Oversight**: Policies that ensure the AI system remains under human control and that its actions are verifiably aligned with human intent and can be overseen by humans.
-   **Technical Robustness and Safety**: Policies related to the system's resilience against errors, faults, or inconsistencies, its accuracy, reliability, and its ability to minimize unintended harm.
-   **Privacy and Data Governance**: Policies concerning data protection in compliance with existing privacy rules, and ensuring high standards of data quality and integrity.
-   **Transparency**: Policies that require the system to be understandable, its operations traceable, and that humans are made aware they are interacting with an AI system.
-   **Diversity, Non-discrimination and Fairness**: Policies that promote equal access, avoid discriminatory impacts, and prevent unfair biases prohibited by law.
-   **Societal and Environmental Well-being**: Policies that ensure the AI system is used in a sustainable and environmentally friendly manner and benefits all human beings, while assessing long-term societal impacts.

--- RISK CATEGORIES ---
<RISK_CATEGORIES>

--- INSTRUCTIONS ---
-   For "ethical_principles", you may select multiple values if applicable.
-   For "risk_categories", you **must choose only one**. Select the category whose LTL template most directly and simply represents the core technical constraint of the policy.

--- INPUT ---
## Refined Policy:
<POLICY>

--- OUTPUT FORMAT (JSON ONLY) ---
Your output must be a single JSON object.

{
  "ethical_principles": ["string"], // A list of all relevant principles from the list above.
  "risk_categories": "string"  // A string the SINGLE most relevant risk category name.
}
"""

def get_policy_classification_prompt(policy: Dict, risk_categories: Dict) -> dict:
    """
    Builds the prompt for classifying a policy with an EU AI Act principle and Risk Categories.
    """
    # Format the risk categories for clear presentation in the prompt
    risk_categories_text = ""
    for _key, value in risk_categories.items():
        risk_categories_text += f"-   **{value['name']}**: {value['description']}\n"

    user_prompt = CLASSIFICATION_USER.replace("<POLICY>", json.dumps(policy, indent=2))
    user_prompt = user_prompt.replace("<RISK_CATEGORIES>", risk_categories_text.strip())

    return {
        "system": CLASSIFICATION_SYSTEM,
        "user": user_prompt,
    }