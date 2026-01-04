# prompts/get_api_mapping_prompt.py
import json
from typing import Dict

API_MAPPING_SYSTEM = """
You are a specialized AI component within a larger application, acting as a **LTL Generation Service**.
Your function is to take a natural language policy, a corresponding LTL template, and API documentation, and then generate a list of final, concrete LTL rules by mapping **only the correct, existing APIs** to the template's placeholders.
**Crucially, you MUST NOT invent or assume any API names. Every API name used in the final LTL rules MUST exist exactly as written in the provided API Documentation.** You must always generate at least one LTL rule.
"""

API_MAPPING_USER = """
The application has processed a policy and determined its LTL template. Your final task is to generate one or more complete LTL rules using **only** the API names provided in the documentation. The list of final LTL rules cannot be empty.

--- REASONING STEPS ---
1.  **Analyze the Policy**: Read the `policy_description` to understand the core constraint.
2.  **Decompose if Necessary**: Determine if the policy contains multiple distinct, testable obligations. If so, plan to create a separate, atomic LTL rule for each one.
3.  **Analyze the LTL Template**: Identify the role of each placeholder (e.g., `P1`, `P2`).
4.  **Consult the API Documentation**: Review the available API functions listed under `<API_DOC>`. **Identify the exact names** of the relevant API calls.
5.  **Map APIs to Placeholders**: For each planned LTL rule, determine which **specific API calls listed in `<API_DOC>`** best represent the concepts in the placeholders.
    * **CRITICAL CONSTRAINT:** You **MUST** find the *best possible* mapping using only the APIs listed in `<API_DOC>`. Do **NOT** invent API names. If no existing API perfectly represents a concept, select the closest available API as a proxy. You **MUST** clearly document any assumptions, limitations, or potential inaccuracies arising from using a proxy API in the `api_mapping_analysis`. If the mapping seems forced or illogical due to the lack of suitable APIs, explicitly state that this might indicate the source policy is too abstract or not well-suited for concrete LTL generation with this specific API set.
6.  **Generate Final Rules**: Substitute the **exact, validated API names from `<API_DOC>`** into the LTL template to create the final logic strings and return them as a list. This list **must not** be empty.

--- INPUTS ---
## 1. Refined Policy:
<POLICY>

## 2. LTL Template Details:
<LTL_TEMPLATE_DETAILS>

## 3. API Documentation:
<API_DOC>

--- OUTPUT FORMAT (JSON ONLY) ---
Your output must be a single JSON object.

{
  "api_mapping_analysis": "string", // Your step-by-step reasoning for how you mapped the APIs (including proxies) to the placeholders for all generated rules. Clearly state assumptions, limitations, and if the source policy seems poorly suited for mapping.
  "final_ltl_rules": ["string"] // A list of final, complete LTL logic strings, using ONLY API names found in <API_DOC>.
}
"""

def get_api_mapping_prompt(policy: Dict, risk_category_obj: Dict, api_doc: Dict) -> dict:
    """
    Builds the prompt for mapping APIs to an LTL template.
    """
    user_prompt = API_MAPPING_USER.replace("<POLICY>", json.dumps(policy, indent=2))
    user_prompt = user_prompt.replace("<LTL_TEMPLATE_DETAILS>", json.dumps(risk_category_obj, indent=2))
    user_prompt = user_prompt.replace("<API_DOC>", json.dumps(api_doc, indent=2))

    return {
        "system": API_MAPPING_SYSTEM,
        "user": user_prompt,
    }