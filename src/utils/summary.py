import os
import json
from collections import defaultdict
from statistics import mean


DEFAULT_VALUES = {
    "partial_resampling": False,
    "continuous_formulation": False,
}


def summarize_metadata(output_dir: str,
                       filter_dict: dict,
                       group_by_list: list,
                       target_attribute: str) -> list[dict]:
    """
    Walks through each subfolder of `output_dir`, loads metadata.json,
    filters by filter_dict, groups by group_by_list attributes,
    and for each group collects target_attribute values and computes
    mean, min, max.

    Returns:
      A list of dicts, one per unique group, each containing:
        - all filter_dict keys and values
        - each group_by_list key and its group value
        - target_attribute: list of all values in that group
        - target_attribute_mean, _min, _max
    """
    # 1) Load and filter
    filtered = []
    for root, dirs, files in os.walk(output_dir):
        if 'metadata.json' not in files:
            continue
        path = os.path.join(root, 'metadata.json')
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        # apply default values for missing keys
        for key, default in DEFAULT_VALUES.items():
            if meta.get(key) is None:
                meta[key] = default
        # apply filter_dict
        if all(meta.get(k) == v for k, v in filter_dict.items()):
            filtered.append(meta)

    # 2) Group
    groups = defaultdict(list)
    for meta in filtered:
        key = tuple(meta.get(attr) for attr in group_by_list)
        groups[key].append(meta)

    # 3) Summarize
    results = []
    for key_vals, metas in groups.items():
        # collect target values (ensure they’re numeric)
        vals = [meta[target_attribute] for meta in metas]
        grp = {attr: val for attr, val in zip(group_by_list, key_vals)}
        entry = {
            **filter_dict,               # include the filter criteria
            **grp,                       # include group‑by attributes
            target_attribute: vals,      # raw list
            f"{target_attribute}_mean": mean(vals),
            f"{target_attribute}_min": min(vals),
            f"{target_attribute}_max": max(vals),
        }
        results.append(entry)

    return results

output_dir = "./output_SMC"

def print_results():
    for result in results:
        # print each field in bold
        for key in group_by:
            print(f"**{key}**: {result[key]}  ")
        print(f"**{target}**: {result[target]}  ")
        print(f"**Average**: {result[f'{target}_mean']:.2f}  ({len(result[target])})")
        # separator between “boxes”
        print("\n---\n")

# Best-of-N
print("# Best-of-N")
filter_dict = {"proposal_type": "without_SMC"}
group_by = ["use_remdm", "CFG", "steps", "reward_name", "prompt", "num_images"]
target = "best_reward"

results = summarize_metadata(output_dir, filter_dict, group_by, target)
print_results()

# SMC with Reverse as proposal
print("# SMC with Reverse as proposal")
filter_dict = {"proposal_type": "reverse"}
group_by = ["use_remdm", "CFG", "steps", "reward_name", "prompt", "num_images", "phi", "tau", "kl_weight", "lambda_tempering", "lambda_one_at", "resample_frequency", "partial_resampling"]
target = "best_reward"
results = summarize_metadata(output_dir, filter_dict, group_by, target)
print_results()

# SMC with Locally Optimal Proposal
print("# SMC with Locally Optimal Proposal")
filter_dict = {"proposal_type": "locally_optimal"}
group_by = ["use_remdm", "CFG", "steps", "reward_name", "prompt", "num_images", "phi", "tau", "kl_weight", "lambda_tempering", "lambda_one_at", "resample_frequency", "partial_resampling", "continuous_formulation"]
target = "best_reward"
results = summarize_metadata(output_dir, filter_dict, group_by, target)
print_results()

# SMC with Locally Optimal Proposal - Straight through grads
print("# SMC with Locally Optimal Proposal - Straight through grads")
filter_dict = {"proposal_type": "straight_through_gradients"}
group_by = ["use_remdm", "CFG", "steps", "reward_name", "prompt", "num_images", "phi", "tau", "kl_weight", "lambda_tempering", "lambda_one_at", "resample_frequency", "partial_resampling"]
target = "best_reward"
results = summarize_metadata(output_dir, filter_dict, group_by, target)
print_results()
