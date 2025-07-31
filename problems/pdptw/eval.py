import sys
import os
import logging
import numpy as np
import inspect
import glob

sys.path.insert(0, "../../../")

from problems.pdptw.verifier import verify
from problems.pdptw.utils import read_instance, calculate_distance

import gpt



def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name

possible_func_names = ["solve", "solve_v1", "solve_v2", "solve_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


def evaluate(input_file: str, solution_file: str) -> float:
    """Cost calculation function: calculates the solution cost.

    Args:
        input_file: Path to the input file containing the instance
        solution_file: Path to the solution file

    Returns:
        float: The cost of the solution, or infinity if invalid
    """
    instance = read_instance(input_file)
    nodes = instance.nodes  # Use the dictionary directly
    total_cost = 0.0

    with open(solution_file, "r") as file:
        lines = file.readlines()
        for line_num, line in enumerate(
            lines[1:], start=1
        ):  # Skip the first line (cost line)
            route = list(map(int, line.strip().split()))
            if (
                not route
                or route[0] != instance.depot_node[0]
                or route[-1] != instance.depot_node[0]
            ):
                raise ValueError(
                    f"Route on line {line_num} must start and end at the depot."
                )

            route_cost = 0.0
            for i in range(len(route) - 1):
                current_node = nodes[route[i]]
                next_node = nodes[route[i + 1]]
                route_cost += calculate_distance(
                    current_node.x, current_node.y, next_node.x, next_node.y
                )

            print(f"Cost for route {line_num}: {route_cost}")
            total_cost += route_cost

    return total_cost

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
        benchmark_dir = os.path.join(basepath, "benchmark")
        results_dir = os.path.join(basepath, "results")
        os.makedirs(results_dir, exist_ok=True)
        input_files = sorted(glob.glob(os.path.join(benchmark_dir, "*.pdptw")))
        objs = []
        for input_file in input_files:
            filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(results_dir, f"{filename}.json")
            obj = solve_main(input_file, output_file)
            objs.append(obj)
        print("[*] Average:")
        if len(objs) > 0:
            print(np.mean(objs))
        else:
            print("No valid results found")

    else:
        raise Exception("Hins debugging")
