# prompts/get_verifiability_refinement_prompt.py
import json
from typing import List, Dict

VR_SYSTEM = """
You are a specialized AI component within a larger application, acting as a **Policy Auditing Service**.
Your function is to analyze raw policy statements and classify them based on how they can be technically verified.
"""

VR_USER = """
The application has extracted the following policies. Your task is to audit this list and classify each policy into one of three categories. Your classification must be extremely strict.

--- POLICY CATEGORIES ---
1.  **structural (Strict)**: A rule *exclusively* about the **temporal order, sequence, co-occurrence, or causal dependency** of API calls. This is *only* about *when* calls can or must happen relative to each other.
    -   *Valid Example*: "Action A must happen *before* Action B."
    -   *Valid Example*: "If Action C occurs, Action D must *also* occur *in the same state*."
    -   *Valid Example*: "Action E can *never* be called."

2.  **value_based**: A rule about the **data content** of an API call's parameters or its return value.
    -   *Example*: "The 'amount' parameter in a transaction must be less than 50."
    -   *Example*: "The 'status' field of the return value from `GetTask` must be 'complete'."

3.  **non_structural_or_abstract**: A rule that is *not* a specific, testable sequence or data constraint. This includes:
    -   Organizational, procedural, or legal duties (e.g., "Reports must be filed quarterly.")
    -   Abstract qualities (e.g., "The system must be secure," "Data must be accurate.")
    -   **Broad feature requirements**: Policies that describe *what* a system must *be able to do*, not the *sequence* of *how* it does it.
    -   *Key Example*: "The system must provide **functionality** for individuals to review their data." (This is a feature requirement, NOT 'structural'.)

--- AUDIT & REASONING STEPS ---
1.  For each policy in the input, analyze its `policy_description`.
2.  Assign it *one* category: `structural`, `value_based`, or `non_structural_or_abstract`.
3.  Provide a brief justification for your classification, **specifically explaining why it is (or is not) a 'structural' (temporal/sequential) rule.**
4.  Construct an output object containing ALL policies that are verifiable (i.e., `structural` or `value_based`), along with your analysis.
5.  **Be very critical**: If a policy describes *what* a system should do (a feature) rather than a *sequence* of actions (an order), it is **non_structural_or_abstract**.

--- INPUT ---
## Extracted Policies:
<INITIAL_POLICIES>

--- OUTPUT FORMAT (JSON ONLY) ---
Your output must be a single JSON object matching this exact structure.

{
  "verified_policies": [
    {
      "policy_type": "structural" | "value_based",
      "refinement_analysis": "string", // Your reasoning for the classification.
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

def get_verifiability_refinement_prompt(initial_policies: List[Dict]) -> dict:
    """
    Builds the prompt for the Verifiability Refinement (VR) step.
    """
    # The input to the prompt should be the raw list of policy dictionaries
    initial_policies_str = json.dumps({"policies": initial_policies}, indent=2)

    return {
        "system": VR_SYSTEM,
        "user": VR_USER.replace("<INITIAL_POLICIES>", initial_policies_str),
    }

