# prompts/get_relevancy_filter_prompt.py
import json
from typing import List, Dict

RF_SYSTEM = """
You are a specialized AI component within a larger application, acting as a **Policy Relevancy Filtering Service**.
Your function is to analyze a list of raw policy statements and filter out any that are not relevant or not testable given a specific API toolkit's documentation.
"""

RF_USER = """
The application has extracted a list of policies. Your first task is to audit this list and filter out any policy that is **not technically testable or applicable** to the provided API toolkit.

--- REASONING STEPS ---
1.  For each policy in the `<INITIAL_POLICIES>` list, analyze its `policy_description`.
2.  Compare the policy's constraint against the tools (and their parameters) listed in the `<API_DOC>`.
3.  **Determine Relevancy**:
    * A policy is **RELEVANT** if its constraint can be clearly mapped to one or more specific API calls, their parameters, or their return values. (e.g., a policy on 'audit logging' is relevant if the API has a `RecordAuditEvent` tool).
    * A policy is **NOT RELEVANT** if it is too abstract (e.g., "The system must be secure") OR if it describes functionality *not present* in the API doc (e.g., a policy about 'blocking a physical card' when the API only has 'transfer funds').
4.  **Construct Output**: Create a list containing *only* the policies you determined are **RELEVANT**.
5.  **Provide Justification**: For each relevant policy, provide a *brief* justification explaining *why* it is relevant to the provided API toolkit.

--- INPUT 1: Extracted Policies ---
<INITIAL_POLICIES>

--- INPUT 2: API Documentation ---
<API_DOC>

--- OUTPUT FORMAT (JSON ONLY) ---
Your output must be a single JSON object matching this exact structure.

{
  "relevant_policies": [
    {
      "relevancy_analysis": "string", // Your justification for why this policy IS relevant.
      "policy": {
        "policy_description": "string",
        "scope": "string",
        "definition": "string",
        "reference": {
          "source_id": "string"
        }
      }
    }
  ]
}
"""

def get_relevancy_filter_prompt(initial_policies: List[Dict], api_doc: Dict) -> dict:
    """
    Builds the prompt for the Relevancy Filtering (RF) step.
    """
    initial_policies_str = json.dumps({"policies": initial_policies}, indent=2)
    api_doc_str = json.dumps(api_doc, indent=2)

    return {
        "system": RF_SYSTEM,
        "user": RF_USER.replace("<INITIAL_POLICIES>", initial_policies_str).replace("<API_DOC>", api_doc_str),
    }