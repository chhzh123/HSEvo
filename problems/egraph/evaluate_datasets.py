#!/usr/bin/env python3
"""
Dataset evaluation script for operator scheduling problem.
Evaluates all JSON datasets in a specified folder and prints results table.
"""

import os
import sys
import glob
import argparse
import numpy as np

# Add the parent directory to the path to import the eval module
sys.path.insert(0, os.path.dirname(__file__))

from eval import solve_main


def print_table(data, headers):
    """
    Print a simple formatted table.
    
    Args:
        data: List of rows (each row is a list of values)
        headers: List of header strings
    """
    if not data:
        return
    
    # Calculate column widths
    col_widths = []
    for i in range(len(headers)):
        max_width = len(headers[i])
        for row in data:
            max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width + 2)  # Add some padding
    
    # Print header
    header_line = "|"
    separator_line = "|"
    for i, header in enumerate(headers):
        header_line += f" {header:<{col_widths[i]}} |"
        separator_line += "-" * (col_widths[i] + 2) + "|"
    
    print(header_line)
    print(separator_line)
    
    # Print data rows
    for row in data:
        row_line = "|"
        for i, cell in enumerate(row):
            row_line += f" {str(cell):<{col_widths[i]}} |"
        print(row_line)


def evaluate_datasets(folder_path: str, output_dir: str = None):
    """
    Evaluate all datasets in the specified folder and all subfolders.
    
    Args:
        folder_path: Path to the folder containing JSON datasets
        output_dir: Directory to save output files (optional)
    
    Returns:
        list: List of tuples (dataset_name, cost)
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []
    
    # Find all JSON files in the folder and all subfolders
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    json_files = sorted(json_files)
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}' or its subfolders")
        return []
    
    print(f"Found {len(json_files)} JSON files in '{folder_path}' and its subfolders")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output files will be saved to '{output_dir}'")
    
    results = []
    
    for input_file in json_files:
        # Get relative path from the input folder to preserve subfolder structure
        rel_path = os.path.relpath(input_file, folder_path)
        dataset_name = os.path.splitext(rel_path)[0]  # Remove .json extension
        
        # Determine output file path
        if output_dir:
            # Preserve subfolder structure in output
            output_file = os.path.join(output_dir, f"{dataset_name}.json")
        else:
            # Use same directory as input file
            output_file = os.path.splitext(input_file)[0] + "_output.out"
        
        print(f"\n[*] Processing dataset: {dataset_name}")
        
        # try:
        cost = solve_main(input_file, output_file)
        results.append((dataset_name, cost))
        print(f"[*] Cost for {dataset_name}: {cost}")
        # except Exception as e:
        #     print(f"[!] Error processing {dataset_name}: {e}")
        #     results.append((dataset_name, "ERROR"))
    
    return results


def print_results_table(results):
    """
    Print results in a formatted table.
    
    Args:
        results: List of tuples (dataset_name, cost)
    """
    if not results:
        print("No results to display.")
        return
    
    # Prepare table data
    table_data = []
    valid_costs = []
    
    for dataset_name, cost in results:
        if isinstance(cost, (int, float)) and cost != float("inf"):
            table_data.append([dataset_name, f"{cost:.2f}"])
            valid_costs.append(cost)
        elif cost == float("inf"):
            table_data.append([dataset_name, "INVALID"])
        else:
            table_data.append([dataset_name, str(cost)])
    
    # Print table
    print("\n" + "="*60)
    print("DATASET EVALUATION RESULTS")
    print("="*60)
    
    headers = ["Dataset Name", "Cost"]
    print_table(table_data, headers)
    
    # Print summary statistics
    if valid_costs:
        print(f"\nSummary Statistics:")
        print(f"  Number of valid solutions: {len(valid_costs)}")
        print(f"  Average cost: {np.mean(valid_costs):.2f}")
        print(f"  Minimum cost: {np.min(valid_costs):.2f}")
        print(f"  Maximum cost: {np.max(valid_costs):.2f}")
        print(f"  Standard deviation: {np.std(valid_costs):.2f}")
    
    invalid_count = sum(1 for _, cost in results if cost == float("inf") or cost == "ERROR")
    if invalid_count > 0:
        print(f"  Number of invalid/error solutions: {invalid_count}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate operator scheduling datasets")
    parser.add_argument("folder_path", help="Path to the folder containing JSON datasets (searches recursively in subfolders)")
    parser.add_argument("--output-dir", "-o", help="Directory to save output files (optional)")
    parser.add_argument("--no-table", action="store_true", help="Skip printing the results table")
    
    args = parser.parse_args()
    
    print(f"[*] Starting evaluation of datasets in: {args.folder_path}")
    
    # Evaluate all datasets
    results = evaluate_datasets(args.folder_path, args.output_dir)
    
    # Print results table
    if not args.no_table:
        print_results_table(results)
    
    print(f"\n[*] Evaluation complete!")


if __name__ == "__main__":
    main() 