# prompts/get_redundancy_pruning_prompt.py
import json
from typing import List, Dict

RP_SYSTEM = """
You are a specialized AI component within a larger application, acting as a **Policy Deduplication Service**.
Your function is to analyze a list of verified technical policies, identify semantic duplicates, and consolidate them into a final, clean list, while aggregating their source references.
"""

RP_USER = """
The application has provided a list of verified policies. Your task is to prune this list by identifying and merging policies that enforce the **same core technical rule**, while aggregating their source references.

--- REASONING STEPS ---
1.  **Identify Core Constraint**: For each policy, first analyze its `policy_description` to determine its fundamental **core technical constraint**. This is the specific rule it enforces (e.g., "Access requires a Request," "Action A must precede Action B," "Sensitive access must be logged").
2.  **Group by Constraint**: Group all policies that enforce the *exact same* core technical constraint. These policies are semantic duplicates, even if their natural language descriptions are different.
3.  **Select Representative**: For each group of duplicates, select the single best-written policy as the "representative". The best one is the most clear, precise, and comprehensive.
4.  **Aggregate Source IDs**: For each group, create a modified representative policy. Take the `source_ids` from the chosen representative policy and all policies in its `merged_policies` list. Combine these IDs into a **JSON list of strings**. Update the `reference.source_ids` field of the representative policy with this new aggregated list.
5.  **Consolidate and Output**: Create a final list containing *only* the unique, representative policies, ensuring each one now has the aggregated `source_ids` list.

--- EXAMPLE OF CONSOLIDATION ---
## Input Policies:
[
  { "policy_description": "All access to patient health information must be logged.", "reference": { "source_ids": ["HIPAA-POLICY-015"] } ... },
  { "policy_description": "The system must log disclosures of PHI.", "reference": { "source_ids": ["AUDIT-RULE-002"] } ... }
]

## Analysis:
- Both policies share the core constraint: (Event: PHI Access) -> (Action: Log).
- The first policy is chosen as the representative.
- The `source_ids` ["HIPAA-POLICY-015"] and ["AUDIT-RULE-002"] are aggregated.

## Final Representative Policy in Output:
{
  "policy_description": "All access to patient health information must be logged.",
  "reference": { "source_ids": ["HIPAA-POLICY-015", "AUDIT-RULE-002"] },
  ...
}

--- GOAL ---
Your goal is to **aggressively prune** this list. It is critical to ensure that **only one policy** represents **each unique technical constraint**, and that its `source_ids` field reflects all original policies that were merged.

--- INPUT ---
## Verified Policies:
<VERIFIED_POLICIES>

--- OUTPUT FORMAT (JSON ONLY) ---
Your output must be a single JSON object with two keys: `deduplication_analysis` (for human review) and `final_policies` (for machine processing).
**Crucially, the `representative_policy` (in `deduplication_analysis`) and the corresponding policy in `final_policies` MUST have their `source_ids` field updated to the new list of strings.**

{
  "deduplication_analysis": [
    {
      "representative_policy": {
        "policy_description": "...",
        "reference": {
          "source_ids": ["ID-001", "ID-005"]
        }
      },
      "justification": "string (Explain why this was chosen and what constraint it represents)",
      "merged_policies": [ { ... policy object with original source_ids ... } ]
    }
  ],
  "final_policies": [
    {
      "policy_description": "...",
      "reference": {
        "source_ids": ["ID-001", "ID-005"]
      }
    }
  ]
}
"""

def get_redundancy_pruning_prompt(verified_policies: List[Dict]) -> dict:
    """
    Builds the prompt for the Redundancy Pruning (RP) step.
    """
    return {
        "system": RP_SYSTEM,
        "user": RP_USER.replace("<VERIFIED_POLICIES>", json.dumps(verified_policies, indent=2)),
    }

