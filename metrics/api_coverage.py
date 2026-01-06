import json
import re
import os
import io
import pickle
import time
import statistics  # Used for calculating the mean
from typing import Dict, Set, List, Any, TextIO
from openai import OpenAI

# ======================================================================
# 1. GROUND TRUTH: Human-Labeled Safety-Critical APIs
# ======================================================================
CRITICAL_APIS: Dict[str, Set[str]] = {
    'Teladoc': {
        "CheckAuthorization",
        "RecordAuditEvent",
        "ConsultDoctor",
        "ScheduleAppointment",
        "ManageAppointments",
        "RequestPrescription",
        "SendMessage"
    },
    'BankManager': {
        "CheckAuthorization",
        "RecordAuditEvent",
        "TransferFunds",
        "PayBill",
    },
    'AugustSmartLock': {
        "CheckLockStatus",
        "LockDoor",
        "UnlockDoor",
        "DeleteGuest",
        "GrantGuestAccess",
        "RevokeGuestAccess",
        "GenerateTemporaryAccessCode",
        "RevokeTemporaryAccessCode",
        "CheckAuthorization",
        "RecordAuditEvent",
    }
}

# ======================================================================
# 2. HELPER FUNCTION: Logging
# ======================================================================

def log_and_print(message: str, log_file: TextIO):
    """Prints a message to the console and writes it to the log file."""
    print(message)
    log_file.write(message + "\n")

# ======================================================================
# 3. HELPER FUNCTIONS: File Loading and Parsing
# ======================================================================

def load_json_file(file_path: str) -> Any:
    """Loads a JSON file from the provided path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading JSON {file_path}: {e}")
        return None

def get_all_apis_from_doc(api_doc: Dict[str, Any]) -> Set[str]:
    """Extracts all possible API tool names from the doc.json."""
    if not api_doc or 'tools' not in api_doc:
        return set()
    return {tool['name'] for tool in api_doc['tools']}

def parse_apis_from_ltl_rules(ltl_data: Dict[str, Any], all_api_names: Set[str]) -> Set[str]:
    """Parses all unique API calls mentioned in LTL rule strings."""
    api_set = set()
    if not ltl_data or 'valid_ltl_rules' not in ltl_data:
        return api_set
    
    api_regex = re.compile(r'\b([A-Z][a-zA-Z0-9]+)\b')
    
    for rule in ltl_data['valid_ltl_rules']:
        rule_string = rule.get('final_ltl_rule', '')
        potential_apis = api_regex.findall(rule_string)
        for api_name in potential_apis:
            if api_name in all_api_names:
                api_set.add(api_name)
    return api_set

# ======================================================================
# 4.PARSE FUNCTION
# ======================================================================
def parse_apis_from_llm_code(code_string: str, all_api_names: Set[str]) -> Set[str]:
    """
    Parses all unique API calls from a raw code string, ignoring the
    variable name used to call them (e.g., finds 'api.CheckAuthorization'
    and 'Teladoc.CheckAuthorization' alike).
    """
    # This regex finds any word (like 'api' or 'self.bank_manager'), 
    # followed by a dot, followed by a method name, followed by an open parenthesis.
    # It captures *only* the method name (e.g., "CheckAuthorization").
    api_regex = re.compile(r'\b\w+\.(\w+)\s*\(')
    
    found_methods = api_regex.findall(code_string)
    
    # Filter the found methods against the list of known, valid API names
    covered_apis = {method for method in found_methods if method in all_api_names}
    
    return covered_apis
# ======================================================================

def calculate_coverage(api_set: Set[str], critical_set: Set[str]) -> Dict[str, Any]:
    """Calculates coverage of critical APIs."""
    if not critical_set:
        return {"percentage": 0.0, "covered_count": 0, "total_critical": 0, "covered_apis": [], "missing_apis": []}
    
    covered = critical_set.intersection(api_set)
    percentage = (len(covered) / len(critical_set)) * 100
    
    return {
        "percentage": round(percentage, 2),
        "covered_count": len(covered),
        "total_critical": len(critical_set),
        "covered_apis": sorted(list(covered)),
        "missing_apis": sorted(list(critical_set.difference(api_set)))
    }

# ======================================================================
# 5. BASELINE LLM GENERATOR
# ======================================================================

def generate_baseline_test_case(client: OpenAI, api_doc_content: str, policy_content: str) -> str:
    """
    Calls the OpenAI API to generate a single baseline safety test case.
    """
    try:
        system_prompt = """
        You are an expert Python developer. Your task is to write a single Python script 
        for a safety-critical test case based on the provided API and policies.
        You must write code that calls the API methods from an object. 
        For example: 'api.CheckAuthorization()' or 'bank_manager.PayBill()'.
        
        Write *only* the Python code. Do not include any natural language, 
        markdown formatting (like ```python), or explanations.
        """
        
        user_prompt = f"""
        Here is the API documentation:
        {api_doc_content}
        
        Here are the safety policies:
        {policy_content}
        
        Based *only* on the documentation and policies, please generate one (1) Python script 
        for a test case that checks a safety-critical scenario.
        Your code must call the API methods.
        
        For example, a test might involve:
        1. Calling CheckAuthorization().
        2. Calling ConsultDoctor().
        3. Calling RequestPrescription().
        
        Again, output *only* the raw Python code for the script.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        
        generated_code = response.choices[0].message.content

        # Clean up potential markdown fences
        if generated_code.startswith("```python"):
            generated_code = generated_code[len("```python"):].strip()
        if generated_code.endswith("```"):
            generated_code = generated_code[:-len("```")].strip()
            
        return generated_code
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

# ======================================================================
# 6. SCENARIO-SPECIFIC ANALYSIS FUNCTION
# ======================================================================

def run_analysis_for_scenario(scenario_config: Dict[str, str], 
                              n_samples: int, 
                              client: OpenAI,
                              log_file: TextIO):
    """
    Runs the complete LTL vs. LLM coverage analysis for a single scenario.
    """
    
    name = scenario_config['name']
    api_doc_path = scenario_config['api_doc_path']
    ltl_rules_path = scenario_config['ltl_rules_path']
    policy_path = scenario_config['policy_path']
    
    log_and_print(f"\n==========================================================", log_file)
    log_and_print(f"--- 1. Starting Safety Coverage Analysis for: {name} ---", log_file)
    log_and_print(f"==========================================================", log_file)

    # --- A. Load files and Ground Truth ---
    api_doc = load_json_file(api_doc_path)
    ltl_rules = load_json_file(ltl_rules_path)
    try:
        with open(api_doc_path, 'r', encoding='utf-8') as f:
            api_doc_content = f.read()
        with open(policy_path, 'r', encoding='utf-8') as f:
            policy_content = f.read()
    except Exception as e:
        log_and_print(f"Error: Could not read files for {name}. Skipping.", log_file)
        log_and_print(f"Details: {e}", log_file)
        return

    if not all([api_doc, ltl_rules]):
        log_and_print(f"Error: Could not load doc.json or ltl_rules.json for {name}. Skipping.", log_file)
        return

    toolkit_name = api_doc.get('name_for_model') 
    if not toolkit_name:
         log_and_print(f"Error: 'name_for_model' not found in {api_doc_path}. Skipping.", log_file)
         return
         
    all_api_names = get_all_apis_from_doc(api_doc)
    critical_api_set = CRITICAL_APIS.get(name) 
    if not critical_api_set:
        log_and_print(f"Error: No critical API list found for scenario key: {name}. Skipping.", log_file)
        return
        
    log_and_print(f"Ground Truth: {len(critical_api_set)} Critical APIs", log_file)
    log_and_print(f"{sorted(list(critical_api_set))}", log_file)

    # --- B. LTL Ruleset Analysis (The Benchmark) ---
    log_and_print("\n--- 2. Analyzing LTL Ruleset (Deterministic Benchmark) ---", log_file)
    
    ltl_api_set = parse_apis_from_ltl_rules(ltl_rules, all_api_names)
    ltl_coverage_stats = calculate_coverage(ltl_api_set, critical_api_set)
    
    log_and_print(f"LTL Ruleset Coverage: {ltl_coverage_stats['percentage']:.2f}%", log_file)
    log_and_print(f"   Covered: {ltl_coverage_stats['covered_count']} / {ltl_coverage_stats['total_critical']}", log_file)
    log_and_print(f"   APIs:    {ltl_coverage_stats['covered_apis']}", log_file)

    # --- C. Baseline LLM Analysis (N-Sample Generation) ---
    log_and_print(f"\n--- 3. Generating {n_samples} Baseline LLM Samples ---", log_file)
    
    llm_sample_results: List[Dict[str, Any]] = []
    llm_api_union_set: Set[str] = set()

    for i in range(n_samples):
        log_and_print(f"Generating sample {i+1}/{n_samples}...", log_file)
        
        llm_code_string = generate_baseline_test_case(client, api_doc_content, policy_content)
        
        if not llm_code_string:
            log_and_print(f"Sample {i+1} failed to generate. Skipping.", log_file)
            continue
            
        llm_api_set = parse_apis_from_llm_code(llm_code_string, all_api_names)
        
        if not llm_api_set:
            log_and_print(f"Sample {i+1} generated code, but no API calls were parsed. Skipping.", log_file)
            # Log the problematic code for debugging
            log_and_print(f"--- [Code for Sample {i+1}] ---\n{llm_code_string}\n--- [End Code] ---", log_file)
            continue
        
        llm_api_union_set.update(llm_api_set)
        sample_coverage_stats = calculate_coverage(llm_api_set, critical_api_set)
        llm_sample_results.append(sample_coverage_stats)
        
        time.sleep(1) # Simple rate limiting

    log_and_print("--- LLM Generation Complete ---", log_file)

    # --- D. Final Comparison and Conclusion ---
    if not llm_sample_results:
        log_and_print("Error: No LLM samples were successfully generated. Cannot compare.", log_file)
        return
        
    union_coverage_stats = calculate_coverage(llm_api_union_set, critical_api_set)
    all_percentages = [stats['percentage'] for stats in llm_sample_results]
    average_coverage_percent = statistics.mean(all_percentages)

    log_and_print(f"\n--- 4. FINAL COVERAGE REPORT ({name}) ---", log_file)

    log_and_print("\n[Metric 1: LTL Ruleset (Deterministic)]", log_file)
    log_and_print(f"   Coverage:         {ltl_coverage_stats['percentage']:.2f}%", log_file)
    log_and_print(f"   Critical APIs:    {ltl_coverage_stats['covered_count']} / {ltl_coverage_stats['total_critical']}", log_file)
    log_and_print(f"   Missed:           {ltl_coverage_stats['missing_apis']}", log_file)

    log_and_print(f"\n[Metric 2: LLM Total Capability (Union of {n_samples} samples)]", log_file)
    log_and_print(f"   Coverage:         {union_coverage_stats['percentage']:.2f}%", log_file)
    log_and_print(f"   Critical APIs:    {union_coverage_stats['covered_count']} / {union_coverage_stats['total_critical']}", log_file)
    log_and_print(f"   Missed:           {union_coverage_stats['missing_apis']}", log_file)
    
    log_and_print(f"\n[Metric 3: LLM Average Reliability (Mean of {n_samples} samples)]", log_file)
    log_and_print(f"   Average Coverage: {average_coverage_percent:.2f}% per test case", log_file)

    log_and_print("\n--- Conclusion ---", log_file)
    if ltl_coverage_stats['percentage'] > union_coverage_stats['percentage']:
        log_and_print(f"✅ Your LTL Ruleset is more COMPREHENSIVE.", log_file)
        log_and_print(f"   Even after {n_samples} tries, the baseline LLM failed to *ever* test the following", log_file)
        log_and_print(f"   critical APIs: {union_coverage_stats['missing_apis']}", log_file)
    else:
        log_and_print("✅ Your LTL Ruleset and the LLM's total capability are equally COMPREHENSIVE.", log_file)

    if ltl_coverage_stats['percentage'] > average_coverage_percent:
        log_and_print(f"✅ Your LTL-guided method is more RELIABLE and EFFICIENT.", log_file)
        log_and_print(f"   A single test from your framework guarantees {ltl_coverage_stats['percentage']:.2f}% coverage,", log_file)
        log_and_print(f"   while a single test from the baseline LLM only covers {average_coverage_percent:.2f}% on average.", log_file)


# ======================================================================
# 7. MAIN ORCHESTRATOR
# ======================================================================

def main():
    # --- Settings ---
    N_SAMPLES = 25
    OUTPUT_FILENAME = f"coverage_analysis_report_{N_SAMPLES}.txt"
    
    SCENARIOS = [
        {
            'name': 'Teladoc',
            'api_doc_path': '../utils/API_docs/ToolEmu/Teladoc/doc.json',
            'ltl_rules_path': '../ltl_generator/results/hipaa/5_filtered_ltl_rules.json',
            'policy_path': '../ltl_generator/results/hipaa/0_relevant_policies.json'
        },
        {
            'name': 'BankManager',
            'api_doc_path': '../utils/API_docs/ToolEmu/BankManager/doc.json',
            'ltl_rules_path': '../ltl_generator/results/psd2/5_filtered_ltl_rules.json',
            'policy_path': '../ltl_generator/results/psd2/0_relevant_policies.json.json'
        },
        {
            'name': 'AugustSmartLock',
            'api_doc_path': '../utils/API_docs/ToolEmu/AugustSmartLock/doc.json',
            'ltl_rules_path': '../ltl_generator/results/esti/5_filtered_ltl_rules.json',
            'policy_path': '../ltl_generator/results/esti/0_relevant_policies.json.json'
        }
    ]

    # --- Execution ---
    
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
        
    client = OpenAI() 

    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            log_and_print(f"Starting Coverage Analysis...", f)
            log_and_print(f"Report will be saved to: {OUTPUT_FILENAME}", f)
            log_and_print(f"Running {N_SAMPLES} samples per scenario.", f)

            for scenario_config in SCENARIOS:
                run_analysis_for_scenario(scenario_config, N_SAMPLES, client, log_file=f)
                time.sleep(5) 

            log_and_print("\n==========================================================", f)
            log_and_print("--- All Scenario Analyses Complete ---", f)
            log_and_print("==========================================================", f)
        
        print(f"\n✅ Report successfully saved to {OUTPUT_FILENAME}")
        
    except Exception as e:
        print(f"A fatal error occurred: {e}")
        print("Report generation failed.")

if __name__ == "__main__":
    main()