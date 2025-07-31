import os
import sys
import logging
import sys
import numpy as np
import inspect
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print(sys.path)

from problems.egraph.verifier import verify, load_graph, load_solution

import gpt


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name


possible_func_names = ["solve", "solve_v1", "solve_v2", "solve_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


def evaluate(input_file: str, output_file: str):
    """Sum the 'cost' field of every selected node that actually exists."""
    nodes, roots = load_graph(input_file)
    selected = load_solution(output_file)
    return sum(nodes[n]["cost"] for n in selected if n in nodes)


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
        benchmark_dir = os.path.join(basepath, "../datasets/egraph/demo")
        results_dir = os.path.join(basepath, "results")
        os.makedirs(results_dir, exist_ok=True)
        input_files = sorted(glob.glob(os.path.join(benchmark_dir, "*.json")))
        objs = []
        for input_file in input_files:
            filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(results_dir, f"{filename}.txt")
            obj = solve_main(input_file, output_file)
            objs.append(obj)
        print("[*] Average:")
        print(np.mean(objs))

    else:
        print("[*] Skip validation ...")
        pass
