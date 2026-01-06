import json
import re
import os
import time
import statistics
from typing import Dict, Set, List, Any, Tuple
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================
# 1. CONFIGURATION
# ======================================================================

LLM_BUDGET = 40  # Number of LLM samples to generate

CRITICAL_APIS: Dict[str, Set[str]] = {
    'Teladoc': {
        "CheckAuthorization", "RecordAuditEvent", "ConsultDoctor",
        "ScheduleAppointment", "ManageAppointments", "RequestPrescription",
        "SendMessage"
    },
    'BankManager': {
        "CheckAuthorization", "RecordAuditEvent", "TransferFunds", "PayBill"
    },
    'AugustSmartLock': {
        "LockDoor", "UnlockDoor", "AddGuest", "DeleteGuest",
        "GrantGuestAccess", "RevokeGuestAccess", "GenerateTemporaryAccessCode",
        "RevokeTemporaryAccessCode", "CheckAuthorization", "RecordAuditEvent"
    }
}

# Paths
RESULTS_DIR = "ltl_generator/results"
ATC_TREND_DIR = "results/ground_truth_data"
BASELINE_OUTPUT_DIR = "results/rq1_baseline_data_gpt5"
os.makedirs(BASELINE_OUTPUT_DIR, exist_ok=True)

# ======================================================================
# 2. METRIC CALCULATORS
# ======================================================================

class ATCCalculator:
    """Calculates Adjacent Transition Coverage (ATC)."""
    def __init__(self, all_possible_apis: List[str]):
        self.unique_adjacent_pairs: Set[Tuple[str, str]] = set()
        self.total_possible_apis = len(all_possible_apis)
        if self.total_possible_apis < 2:
            self.max_possible_pairs = 0
        else:
            self.max_possible_pairs = self.total_possible_apis * self.total_possible_apis

    def add_trace(self, api_call_trace: List[str]):
        if len(api_call_trace) < 2: return 
        for i in range(len(api_call_trace) - 1):
            pair = (api_call_trace[i], api_call_trace[i+1])
            self.unique_adjacent_pairs.add(pair)

    def calculate_atc(self) -> float:
        if self.max_possible_pairs == 0: return 0.0
        return len(self.unique_adjacent_pairs) / self.max_possible_pairs

class CriticalCoverageCalculator:
    """Calculates percentage of Critical APIs covered."""
    def __init__(self, critical_set: Set[str]):
        self.critical_set = critical_set
        self.covered_set = set()
        
    def add_trace(self, api_call_trace: List[str]):
        for api in api_call_trace:
            if api in self.critical_set:
                self.covered_set.add(api)
                
    def get_coverage_percentage(self) -> float:
        if not self.critical_set: return 0.0
        return (len(self.covered_set) / len(self.critical_set)) * 100.0
    
    def get_missing_apis(self) -> List[str]:
        return sorted(list(self.critical_set - self.covered_set))

# ======================================================================
# 3. HELPER FUNCTIONS
# ======================================================================

def load_json_file(file_path: str) -> Any:
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return None

def count_words(text: str) -> int:
    if not text: return 0
    return len(re.findall(r'\w+', text))

def get_list_count(data: Any, key: str = None) -> int:
    if data is None: return 0
    if isinstance(data, list): return len(data)
    if isinstance(data, dict):
        if key and key in data: return len(data[key])
        for k in ["policies", "items", "relevant_policies", "final_ltl_rules", "valid_ltl_rules"]:
            if k in data: return len(data[k])
    return 0

def get_all_apis_from_doc(api_doc: Dict[str, Any]) -> List[str]:
    if not api_doc or 'tools' not in api_doc: return []
    return [tool['name'] for tool in api_doc['tools']]

def extract_trace_from_code(code_string: str, all_api_names: List[str]) -> List[str]:
    # Regex for .MethodName(
    api_regex = re.compile(r'\b\w+\.(\w+)\s*\(')
    found_methods = api_regex.findall(code_string)
    valid_set = set(all_api_names)
    return [m for m in found_methods if m in valid_set]

def extract_trace_from_ltl(ltl_string: str, all_api_names: List[str]) -> List[str]:
    potential_apis = re.findall(r'\b([A-Z][a-zA-Z0-9]+)\b', ltl_string)
    valid_set = set(all_api_names)
    return [m for m in potential_apis if m in valid_set]

def load_fuzzer_trend(scenario_name: str) -> List[Dict[str, float]]:
    file_map = {
        "Bank Manager": "bank_manager_atc_trend.json",
        "Teladoc": "teladoc_atc_trend.json",
        "Smart Lock": "smart_lock_atc_trend.json"
    }
    filename = file_map.get(scenario_name)
    path = os.path.join(ATC_TREND_DIR, filename)
    data = load_json_file(path)
    if not data: return []

    series = []
    sorted_keys = sorted(data.keys(), key=lambda k: int(k))
    for k in sorted_keys:
        series.append({"x": int(k), "atc": float(data[k])})
    return series

# ======================================================================
# 4. BENCHMARK DATA EXTRACTION (NEW)
# ======================================================================

def get_benchmark_metrics(scenario_config: Dict, all_apis: List[str], critical_set: Set[str]) -> Tuple[List[Dict], float, float]:
    """
    Extracts traces for the 40 labeled benchmark items and calculates metrics.
    """
    labeled_path = scenario_config['labeled_data_path']
    ground_truth_path = scenario_config['ground_truth_path']
    
    labeled_data = load_json_file(labeled_path)
    ground_truth_data = load_json_file(ground_truth_path)
    
    if not labeled_data or not ground_truth_data:
        print(f"   [Error] Missing labeled or ground truth data for {scenario_config['name']}")
        return [], 0.0, 0.0

    # 1. Identify Selected Trace IDs
    # Handle list vs dict structure in labeled data
    prompts = labeled_data if isinstance(labeled_data, list) else labeled_data.get("nl_prompt_pairs", [])
    selected_ids = {p.get("trace_id") for p in prompts if p.get("trace_id")}
    
    print(f"   [Benchmark] Found {len(selected_ids)} selected traces in labeled dataset.")

    # 2. Extract Code for these IDs from Ground Truth
    trace_map = {}
    gt_cases = ground_truth_data.get("test_cases", [])
    for tc in gt_cases:
        tid = tc.get("trace_id")
        if tid in selected_ids:
            # Use 'generated_program' as the source of truth for the trace
            trace_map[tid] = tc.get("generated_program", "")

    # 3. Calculate Metrics
    atc_calc = ATCCalculator(all_apis)
    crit_calc = CriticalCoverageCalculator(critical_set)
    series = []
    
    # Process in the order they appear in labeled data (preserves intended curriculum if any)
    count = 0
    for tid in selected_ids:
        code = trace_map.get(tid, "")
        if not code: continue
        
        trace = extract_trace_from_code(code, all_apis)
        atc_calc.add_trace(trace)
        crit_calc.add_trace(trace)
        
        count += 1
        series.append({"x": count, "atc": atc_calc.calculate_atc()})

    return series, atc_calc.calculate_atc(), crit_calc.get_coverage_percentage()

# ======================================================================
# 5. LLM BASELINE GENERATION
# ======================================================================

def generate_baseline_test_case(client: OpenAI, api_doc_content: str, policy_content: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini", 
            messages=[
                {"role": "system", "content": "You are a QA Engineer. Write a Python test script using the provided API to verify a specific safety policy. Vary your scenario."},
                {"role": "user", "content": f"API Doc:\n{api_doc_content}\n\nPolicies:\n{policy_content}\n\nGenerate 1 Python test case."}
            ],
        )
        content = response.choices[0].message.content
        if "```python" in content:
            content = content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        return content
    except Exception:
        return ""

def get_llm_baseline_data(scenario_config: Dict, client: OpenAI, all_apis: List[str], critical_set: Set[str]) -> Tuple[List[Dict], float, float]:
    internal_name = scenario_config['internal_name']
    raw_code_path = os.path.join(BASELINE_OUTPUT_DIR, f"{internal_name}_llm_baseline_raw.json")
    
    llm_atc_calc = ATCCalculator(all_apis)
    llm_crit_calc = CriticalCoverageCalculator(critical_set)
    series = []
    raw_codes = []

    if os.path.exists(raw_code_path):
        print(f"   [Cache Hit] Loading LLM traces from {raw_code_path}...")
        existing_data = load_json_file(raw_code_path)
        for entry in existing_data:
            trace = entry.get("extracted_trace", [])
            llm_atc_calc.add_trace(trace)
            llm_crit_calc.add_trace(trace)
            series.append({"x": entry["iteration"], "atc": llm_atc_calc.calculate_atc()})
        return series, llm_atc_calc.calculate_atc(), llm_crit_calc.get_coverage_percentage()

    print(f"   [Cache Miss] Generating {LLM_BUDGET} new LLM samples...")
    api_doc = load_json_file(scenario_config['api_doc_path'])
    with open(scenario_config['api_doc_path'], 'r') as f: api_doc_str = f.read()

    for i in range(LLM_BUDGET):
        print(f"\r   Generation {i+1}/{LLM_BUDGET}...", end="")
        code = generate_baseline_test_case(client, api_doc_str)
        trace = extract_trace_from_code(code, all_apis)
        
        llm_atc_calc.add_trace(trace)
        llm_crit_calc.add_trace(trace)
        
        series.append({"x": i + 1, "atc": llm_atc_calc.calculate_atc()})
        raw_codes.append({"iteration": i + 1, "code": code, "extracted_trace": trace})
        time.sleep(0.1)
    
    print("\n   Saving raw codes...")
    with open(raw_code_path, 'w') as f: json.dump(raw_codes, f, indent=2)
    return series, llm_atc_calc.calculate_atc(), llm_crit_calc.get_coverage_percentage()

def get_fuzzer_critical_coverage(scenario_config: Dict, all_apis: List[str], critical_set: Set[str]) -> float:
    ltl_data = load_json_file(scenario_config['ltl_rules_path'])
    crit_calc = CriticalCoverageCalculator(critical_set)
    if ltl_data:
        for rule in ltl_data.get('valid_ltl_rules', []):
            trace = extract_trace_from_ltl(rule.get('final_ltl_rule', ''), all_apis)
            crit_calc.add_trace(trace)
    return crit_calc.get_coverage_percentage()

# ======================================================================
# 6. ANALYSIS & PLOTTING
# ======================================================================

def run_analysis_for_scenario(scenario_config: Dict[str, str], client: OpenAI):
    name = scenario_config['name']
    internal_name = scenario_config['internal_name']
    print(f"\n--> Processing Scenario: {name}")

    api_doc = load_json_file(scenario_config['api_doc_path'])
    all_apis = get_all_apis_from_doc(api_doc)
    critical_set = CRITICAL_APIS.get(internal_name, set())

    # 1. Full Fuzzer Metrics (Green)
    fuzzer_series = load_fuzzer_trend(name)
    fuzzer_final_atc = fuzzer_series[-1]['atc'] if fuzzer_series else 0
    fuzzer_crit_cov = get_fuzzer_critical_coverage(scenario_config, all_apis, critical_set)

    # 2. Benchmark Metrics (Blue - Selected 40)
    bench_series, bench_atc, bench_crit = get_benchmark_metrics(scenario_config, all_apis, critical_set)

    # 3. LLM Baseline Metrics (Red)
    llm_series, llm_final_atc, llm_crit_cov = get_llm_baseline_data(scenario_config, client, all_apis, critical_set)

    # 4. Save Metrics
    metrics = {
        "scenario": name,
        "fuzzer_full": {
            "samples": len(fuzzer_series),
            "final_atc": fuzzer_final_atc,
            "critical_coverage": fuzzer_crit_cov,
            "trend": fuzzer_series
        },
        "benchmark_selected": {
            "samples": len(bench_series),
            "final_atc": bench_atc,
            "critical_coverage": bench_crit,
            "trend": bench_series
        },
        "llm_baseline": {
            "samples": LLM_BUDGET,
            "final_atc": llm_final_atc,
            "critical_coverage": llm_crit_cov,
            "trend": llm_series
        }
    }
    
    metric_path = os.path.join(BASELINE_OUTPUT_DIR, f"{internal_name}_rq1_metrics.json")
    with open(metric_path, 'w') as f: json.dump(metrics, f, indent=4)

    # 5. Plot Growth Comparison
    plot_comparison(llm_series, fuzzer_series, bench_series, name, internal_name)

def plot_comparison(llm_data, fuzzer_data, bench_data, scenario_name, filename_suffix):
    plt.figure(figsize=(10, 6))
    
    # # Fuzzer (Green - Long)
    # fx = [d['x'] for d in fuzzer_data]
    # fy = [d['atc'] for d in fuzzer_data]
    # plt.plot(fx, fy, label='Full Fuzzer (All Generated)', color='#2ecc71', linewidth=2, alpha=0.6)
    
    # Benchmark (Blue - Selected 40)
    bx = [d['x'] for d in bench_data]
    by = [d['atc'] for d in bench_data]
    plt.plot(bx, by, label='Benchmark (Selected 40)', color='#3498db', linewidth=2.5)

    # LLM (Red - Baseline)
    lx = [d['x'] for d in llm_data]
    ly = [d['atc'] for d in llm_data]
    plt.plot(lx, ly, label=f'LLM Baseline (Budget={len(lx)})', color='#e74c3c', linewidth=2.5, linestyle='--')
    
    if lx:
        plt.scatter([lx[-1]], [ly[-1]], color='#e74c3c', s=80, zorder=5)

    plt.title(f"Diversity Growth: {scenario_name}")
    plt.xlabel("Number of Traces")
    plt.ylabel("Adjacent Transition Coverage (ATC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Ensure X axis covers the benchmark length
    max_x = max(len(lx), len(bx))
    plt.xlim(0, max_x * 1.05)
    
    save_path = os.path.join(BASELINE_OUTPUT_DIR, f"{filename_suffix}_atc_growth.png")
    plt.savefig(save_path, dpi=300)
    print(f"   [Chart] Saved to {save_path}")
    plt.close()

# ======================================================================
# 7. MAIN
# ======================================================================

def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY not set.")
        return

    client = OpenAI()

    SCENARIOS = [
        {
            "name": "Bank Manager", "internal_name": "BankManager",
            "policy_file": "psd2", "dir": "psd2",
            "api_doc_path": "utils/API_docs/ToolEmu/BankManager/doc.json",
            "ltl_rules_path": "ltl_generator/results/psd2/7_label_ltl_rules.json",
            "labeled_data_path": "results/benchmark_data/labeled_bank_manager_dataset.json",
            "ground_truth_path": "results/ground_truth_data/bank_manager_ground_truth_cases.json"
        },
        {
            "name": "Teladoc", "internal_name": "Teladoc",
            "policy_file": "hipaa", "dir": "hipaa",
            "api_doc_path": "utils/API_docs/ToolEmu/Teladoc/doc.json",
            "ltl_rules_path": "ltl_generator/results/hipaa/7_label_ltl_rules.json",
            "labeled_data_path": "results/benchmark_data/labeled_teladoc_dataset.json",
            "ground_truth_path": "results/ground_truth_data/teladoc_ground_truth_cases.json"
        },
        {
            "name": "Smart Lock", "internal_name": "AugustSmartLock",
            "policy_file": "esti", "dir": "esti",
            "api_doc_path": "utils/API_docs/ToolEmu/AugustSmartLock/doc.json",
            "ltl_rules_path": "ltl_generator/results/esti/7_label_ltl_rules.json",
            "labeled_data_path": "results/benchmark_data/labeled_smart_lock_dataset.json",
            "ground_truth_path": "results/ground_truth_data/smart_lock_ground_truth_cases.json"
        }
    ]
    
    for sc in SCENARIOS:
        run_analysis_for_scenario(sc, client)

if __name__ == "__main__":
    main()