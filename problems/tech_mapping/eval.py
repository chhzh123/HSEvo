import os
import sys
import logging
import sys
import numpy as np
import inspect
import glob

sys.path.insert(0, "../../../")

from problems.tech_mapping.verifier import verify
from problems.tech_mapping.evaluator import evaluate

import gpt

def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name

possible_func_names = ["solve", "solve_v1", "solve_v2", "solve_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


def solve_main(input_file: str, output_file: str):
    print(f"[*] Input file: {input_file}")
    heuristics(input_file, output_file)
    print(f"[*] Output file: {output_file}")
    is_valid, error_message = verify(input_file, output_file)
    if not is_valid:
        # return float("inf")
        return 20000 # a big number
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
        benchmark_dir = os.path.join(basepath, "benchmark")
        results_dir = os.path.join(basepath, "results")
        os.makedirs(results_dir, exist_ok=True)
        input_files = sorted(glob.glob(os.path.join(benchmark_dir, "*.blif")))
        objs = []
        for input_file in input_files:
            filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(results_dir, f"{filename}.blif")
            obj = solve_main(input_file, output_file)
            objs.append(obj)
        print("[*] Average:")
        print(np.mean(objs))
        # print("Each obj:")
        # print(objs)

    else:
        for problem_size in [20, 50, 100]:
            dataset_path = os.path.join(
                basepath, f"dataset/{mood}{problem_size}_dataset.npy"
            )
            dataset = np.load(dataset_path)
            demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]

            n_instances = node_positions.shape[0]
            logging.info(f"[*] Evaluating {dataset_path}")

            objs = []
            for i, (node_pos, demand) in enumerate(zip(node_positions, demands)):
                obj = solve(node_pos, demand)
                objs.append(obj.item())

            print(f"[*] Average for {problem_size}: {np.mean(objs)}")
