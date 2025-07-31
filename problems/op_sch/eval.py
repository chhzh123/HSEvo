import os
import sys
import logging
import sys
import numpy as np
import inspect
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print(sys.path)

from problems.op_sch.utils import parse_json, parse_schedule
from problems.op_sch.verifier import verify

import gpt

def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name

possible_func_names = ["solve", "solve_v1", "solve_v2", "solve_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


def evaluate(input_file: str, output_file: str) -> int:
    """Cost calculation function: calculates the final latency.

    Args:
        input_file: Path to the input file containing graph and constraints
        output_file: Path to the schedule file containing node start times

    Returns:
        int: The final latency of the schedule

    Final latency is defined as the maximum over operations of (start cycle + delay).
    """
    nodes, delay, _ = parse_json(input_file)
    schedule = parse_schedule(output_file)

    latency = 0
    for node_id, node in nodes.items():
        finish_time = schedule[node_id] + delay[node.resource]
        latency = max(latency, finish_time)
    return latency


def solve_main(input_file: str, output_file: str):
    print(f"[*] Input file: {input_file}")
    heuristics(input_file, output_file)
    print(f"[*] Output file: {output_file}")
    is_valid, error_message = verify(input_file, output_file)
    if not is_valid:
        return float("inf")
    cost = evaluate(input_file, output_file)
    return cost


if __name__ == "__main__":
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ["train", "val"]

    basepath = os.path.dirname(__file__)

    if mood == "train":
        print(f"[*] Dataset loaded.")
        benchmark_dir = os.path.join(basepath, "datasets/operator_scheduling/demo")
        results_dir = os.path.join(basepath, "results")
        os.makedirs(results_dir, exist_ok=True)
        input_files = sorted(glob.glob(os.path.join(benchmark_dir, "*.json")))
        objs = []
        for input_file in input_files:
            filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(results_dir, f"{filename}.json")
            obj = solve_main(input_file, output_file)
            objs.append(obj)
        print("[*] Average:")
        print(np.mean(objs))

    else:
        print("[*] Skip validation ...")
        pass
