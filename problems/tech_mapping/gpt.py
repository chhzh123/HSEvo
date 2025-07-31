import sys
from collections import deque

class Node:
    """Represents a logic gate in the Boolean network."""
    __slots__ = ('name', 'fanin_names', 'fanins', 'fanouts', 'patterns_one', 'patterns_zero', 'const_val')

    def __init__(self, name):
        self.name = name
        self.fanin_names = []  # List of names of direct fanin nodes
        self.fanins = []       # List of Node objects for direct fanins
        self.fanouts = []      # List of Node objects for direct fanouts
        self.patterns_one = [] # Truth table patterns that yield output 1
        self.patterns_zero = []# Truth table patterns that yield output 0
        self.const_val = None  # None, 0, or 1 if the node is a constant gate

def parse_blif(input_file: str):
    """Parses a BLIF file and constructs the Boolean network graph."""
    raw_lines = open(input_file).read().splitlines()
    lines = []
    # Pre-process lines to handle multi-line directives (backslash '\')
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        if line.endswith('\\'):
            merged = line[:-1].rstrip()
            i += 1
            while i < len(raw_lines):
                nxt = raw_lines[i].strip()
                if nxt.endswith('\\'):
                    merged += ' ' + nxt[:-1].rstrip()
                    i += 1
                else:
                    merged += ' ' + nxt
                    i += 1
                    break
            lines.append(merged)
        else:
            lines.append(line)
            i += 1

    model_name = ''
    input_list = []
    output_list = []
    nodes = {} # Dictionary to store Node objects by name

    idx = 0
    while idx < len(lines):
        l = lines[idx].strip()
        if not l: # Empty line
            idx += 1
            continue
        
        parts = l.split()
        directive = parts[0]

        if directive == '.model':
            if len(parts) > 1:
                model_name = parts[1]
            idx += 1
        elif directive == '.inputs':
            for name in parts[1:]:
                input_list.append(name)
                if name not in nodes:
                    nodes[name] = Node(name)
            idx += 1
        elif directive == '.outputs':
            for name in parts[1:]:
                output_list.append(name)
            idx += 1
        elif directive == '.names':
            in_names = parts[1:-1]
            out_name = parts[-1]

            if out_name not in nodes:
                nodes[out_name] = Node(out_name)
            node = nodes[out_name]
            node.fanin_names = in_names[:]

            patterns = []
            j = idx + 1
            while j < len(lines):
                lj = lines[j].strip()
                # Stop if it's an empty line, comment, or another BLIF directive
                if not lj or lj.startswith('.') or lj.startswith('#'):
                    break
                patterns.append(lj)
                j += 1
            
            # Handle constant gates
            if not in_names: # No inputs means a constant gate
                if patterns and patterns[0].strip() == '1':
                    node.const_val = 1
                else:
                    node.const_val = 0
            else:
                for pline in patterns:
                    parts2 = pline.split()
                    mask = parts2[0]
                    bit = '1' if len(parts2) == 1 else parts2[1]
                    if bit == '1':
                        node.patterns_one.append(mask)
                    else:
                        node.patterns_zero.append(mask)
            idx = j
        elif directive == '.end':
            idx += 1
            break
        else: # Unknown directive or malformed line, skip
            idx += 1
            
    # Link fanin/fanout Node objects
    for node in nodes.values():
        node.fanins = []
        for fn in node.fanin_names:
            if fn in nodes: # Only link if the fanin node exists (could be external if incomplete graph)
                node.fanins.append(nodes[fn])
                
    for node in nodes.values():
        for f in node.fanins:
            f.fanouts.append(node)
            
    # Prune unreachable nodes (optimization)
    reachable = set()
    q = deque()
    for out in output_list:
        if out in nodes:
            q.append(out)
            
    while q:
        u_name = q.pop()
        if u_name in reachable:
            continue
        reachable.add(u_name)
        for f_node in nodes[u_name].fanins:
            if f_node.name not in reachable:
                q.append(f_node.name)
                
    # Filter out unreachable nodes
    nodes = {name: node for name, node in nodes.items() if name in reachable}
    # Re-build fanin/fanout lists after pruning
    for node in nodes.values():
        node.fanins = [f for f in node.fanins if f.name in nodes]
        node.fanouts = [f for f in node.fanouts if f.name in nodes]
        
    return model_name, input_list, output_list, nodes

def topological_sort(nodes: dict):
    """Performs a topological sort on the graph."""
    indeg = {name: len(node.fanins) for name, node in nodes.items()}
    q = deque([name for name, d in indeg.items() if d == 0]) # Start with nodes with 0 in-degree (PIs)
    topo_names = []
    
    while q:
        u_name = q.popleft()
        topo_names.append(u_name)
        for w_node in nodes[u_name].fanouts:
            indeg[w_node.name] -= 1
            if indeg[w_node.name] == 0:
                q.append(w_node.name)
                
    # Convert sorted names to Node objects
    topo_list = [nodes[name] for name in topo_names]
    return topo_list

def prune_cuts(cuts_set):
    """Removes redundant (non-minimal) cuts from a set of cuts."""
    # A cut `c` is redundant if there exists another cut `d` in the set
    # such that `d` is a proper subset of `c`.
    cuts = list(cuts_set)
    res = []
    for i in range(len(cuts)):
        c_i = cuts[i]
        skip = False
        for j in range(len(cuts)):
            if i != j:
                c_j = cuts[j]
                if c_j.issubset(c_i) and c_j != c_i: # c_j is a proper subset of c_i
                    skip = True
                    break
        if not skip:
            res.append(c_i)
    return res

def solve_v2(input_file: str, solution_file: str, K: int = 6):
    """
    Solves the K-LUT technology mapping problem to minimize area (number of LUTs).
    Implements a Flow-Map like dynamic programming approach.
    """
    model_name, input_list, output_list, nodes = parse_blif(input_file)
    topo_list = topological_sort(nodes)

    # Dictionary to store K-feasible cuts for each node.
    # cuts[node_name] = list of frozensets, where each frozenset is a set of input
    #                                       names for a K-LUT implementing 'node_name'.
    cuts = {} 

    # Dynamic programming arrays for cost (area) and best cut selection
    cost = {}       # cost[node_name] = minimum LUTs to implement logic rooted at node_name
    best_cut = {}   # best_cut[node_name] = frozenset of input names for the optimal LUT at node_name
    
    INF = 10**18

    # Step 1: Enumerate K-feasible cuts and compute minimum cost for each node
    for n in topo_list:
        node_name = n.name

        if node_name in input_list: # Primary Input: no LUT needed, cost is 0
            cuts[node_name] = [frozenset([node_name])] # Treat PI as its own cut for propagation
            cost[node_name] = 0
            best_cut[node_name] = None
            continue

        # For internal nodes and primary outputs:
        
        # --- Cut Enumeration for 'n' ---
        # `cuts_n_candidates` will hold all K-feasible sets of inputs for LUTs
        # that can implement node 'n'.
        cuts_n_candidates = set()

        # Add the 'trivial' cut: node 'n' itself is the root of a new LUT,
        # with its direct fanins as inputs.
        if len(n.fanin_names) <= K:
            cuts_n_candidates.add(frozenset(n.fanin_names))
        
        # Combine cuts from fanins to form larger K-feasible cones.
        # This approach takes cuts from each fanin 'f' and unions their inputs.
        # `aggregated_cuts` tracks the accumulated cuts from processed fanins.
        aggregated_cuts = None
        for f_node in n.fanins:
            f_cuts = cuts[f_node.name] # Cuts available for fanin f_node
            
            if aggregated_cuts is None:
                # Initialize with cuts from the first fanin
                aggregated_cuts = set(f_cuts)
            else:
                new_aggregated_set = set()
                for c1 in aggregated_cuts: # Combined cuts from previous fanins
                    for c2 in f_cuts:     # Cuts for current fanin
                        union_cut = c1 | c2 # Union their inputs
                        if len(union_cut) <= K:
                            new_aggregated_set.add(union_cut)
                aggregated_cuts = new_aggregated_set
            
        if aggregated_cuts: # If any aggregated cuts were found
            cuts_n_candidates.update(aggregated_cuts)
        
        # Final pruning of cuts for 'n' (remove non-minimal cuts)
        cuts[node_name] = prune_cuts(cuts_n_candidates)

        # --- Cost Calculation for 'n' ---
        # Find the best K-feasible cut that minimizes total area.
        min_cost_n = INF
        optimal_cut_n = None

        if not cuts[node_name]: # Should ideally not happen if graph is valid and K-feasible mappings exist
             # Fallback: if no K-feasible cut found, it implies it cannot be mapped.
             # In practice, for a real circuit, this means partitioning might be required
             # or the problem implies a solution is always feasible.
             # For this problem, we assume mapping is always possible to K-LUTs.
             # This section could be expanded for more robust error handling or advanced mapping.
            raise ValueError(f"Node {node_name} has no K-feasible cuts found with K={K}. Unmappable circuit fragment.")


        for c in cuts[node_name]: # Iterate through all K-feasible cuts for node 'n'
            current_lut_cost = 1 # Cost for this specific LUT
            for m_name in c: # For each input 'm' to this LUT
                # Add the cost to implement 'm' if 'm' is not a primary input
                # The cost of PIs (from `cost` dict) is 0.
                current_lut_cost += cost.get(m_name, 0) 
            
            if current_lut_cost < min_cost_n:
                min_cost_n = current_lut_cost
                optimal_cut_n = c

        cost[node_name] = min_cost_n
        best_cut[node_name] = optimal_cut_n


    # Step 2: Backtrack from primary outputs to identify selected LUTs
    mapped_lut_roots = set() # Set of node names that are roots of selected LUTs
    q = deque()

    # Add primary outputs that are internal nodes (not PIs) to the queue
    for out_name in output_list:
        if out_name in best_cut and best_cut[out_name] is not None:
            q.append(out_name)
    
    while q:
        u_name = q.pop()
        if u_name in mapped_lut_roots:
            continue
        
        mapped_lut_roots.add(u_name)
        
        # Get the optimal cut (inputs) for this LUT
        chosen_cut_inputs = best_cut[u_name]
        
        if chosen_cut_inputs:
            for input_node_name in chosen_cut_inputs:
                # If an input 'm' to the current LUT is itself an internal node (not a PI, not a constant),
                # it needs to be implemented by another LUT (or is a PI that was already available).
                if input_node_name in nodes and \
                   input_node_name not in input_list and \
                   nodes[input_node_name].const_val is None:
                    q.append(input_node_name)


    # Step 3: Generate the output BLIF file with mapped K-LUTs
    with open(solution_file, 'w') as f:
        f.write(f'.model {model_name}\n')
        f.write(f'.inputs {" ".join(input_list)}\n')
        f.write(f'.outputs {" ".join(output_list)}\n')

        # Filter and sort the mapped_lut_roots based on topological order
        # This ensures correct simulation order when generating truth tables.
        mapping_topo_nodes = [n for n in topo_list if n.name in mapped_lut_roots]

        for n_node in mapping_topo_nodes:
            # `leaves` are the input nodes to this specific K-LUT
            leaves = best_cut[n_node.name]
            inputs = sorted(list(leaves)) # Ensure stable input order for truth table

            num_inputs_k = len(inputs)
            num_patterns_N = 1 << num_inputs_k # 2^k possible input combinations

            # Represent each input combination as a bitmask for simulation
            # E.g., for k=2 inputs (A, B): input patterns are 00, 01, 10, 11
            # A_mask = 0101 (for patterns 00,01,10,11)
            # B_mask = 0011 (for patterns 00,01,10,11)
            input_bitmasks = [] # List of bitmasks, one for each input variable position
            for i in range(num_inputs_k):
                mask = 0
                for j in range(num_patterns_N):
                    if (j >> i) & 1: # If the i-th bit of pattern j is 1
                        mask |= (1 << j) # Set the j-th bit in this input variable's mask
                input_bitmasks.append(mask)

            # full_mask is used for bitwise NOT operations (all 1s)
            full_mask = (1 << num_patterns_N) - 1

            # `node_values_in_cone` stores the computed output bitmask for each node in the cone
            # (which contains `n_node` and its predecessors feeding into it, up to `leaves`).
            node_values_in_cone = {} 
            
            # Initialize values for the LUT's inputs (`leaves`)
            for i, input_name in enumerate(inputs):
                node_values_in_cone[input_name] = input_bitmasks[i]
            
            # Identify all nodes within the cone that roots at `n_node` and has `leaves` as inputs.
            # Perform a BFS/DFS from `n_node` backwards, stopping at `leaves` or PIs.
            current_cone_nodes_q = deque([n_node.name])
            visited_cone_nodes = set()
            sorted_cone_nodes_topo = [] # Will be nodes in cone, in topological order

            # Collect nodes in cone AND sort them topologically
            # We must simulate internal nodes within the cone in correct topological order.
            # Start a fresh topological sort limited to the cone if needed,
            # or simply filter the global topo_list
            for node_in_global_topo in topo_list:
                node_name_in_global_topo = node_in_global_topo.name
                if node_name_in_global_topo == n_node.name or \
                   (node_name_in_global_topo in node_values_in_cone and node_name_in_global_topo not in leaves):
                    # Check if this node is relevant to the cone.
                    # This check needs to be smarter: A node `x` is in the cone rooted at `n_node`
                    # with inputs `leaves` if `n_node` depends on `x`, and `x` does not depend
                    # on any other node that is outside the cone or not a primary input/a leaf.

                    # More accurate way to find cone nodes:
                    # Collect all direct predecessors of the LUT root `n_node`.
                    # Then recursively collect predecessors of those, until reaching `leaves` or PIs.
                    if node_name_in_global_topo == n_node.name:
                        # Add n_node itself if not already processed as an input
                        if node_name_in_global_topo not in node_values_in_cone:
                            node_values_in_cone[node_name_in_global_topo] = 0 # Placeholder
                            
                    # Iterate predecessors of n_node until all paths from `leaves` to `n_node` are covered
                    # This check is simpler if we filter nodes from `topo_list`
                    # based on if they are predecessors of `n_node` AND are themselves `leaves` OR are predecessors of `n_node` that are not `leaves` but depend on other non-`leaves` within cone
                    
            # A more robust approach for `cone_nodes`
            cone_nodes_set = set()
            q_cone_identification = deque([n_node.name])
            
            while q_cone_identification:
                curr_node_name = q_cone_identification.popleft()
                if curr_node_name in cone_nodes_set or curr_node_name in leaves:
                    continue
                cone_nodes_set.add(curr_node_name)
                for f_node in nodes[curr_node_name].fanins:
                    # Add fanins if they are not leaves and not already processed
                    if f_node.name not in leaves and f_node.name not in visited_cone_nodes:
                        q_cone_identification.append(f_node.name)
                        visited_cone_nodes.add(f_node.name) # Use visited_cone_nodes to prevent re-adding to queue

            # Now, filter topo_list to get only the nodes that are truly part of this cone and need simulation
            # (excluding the leaf inputs themselves, which are handled).
            sorted_cone_nodes_topo = [nodes[name] for name in topo_list if name in cone_nodes_set]


            # Simulate logic for each node within the cone to get its truth table bitmask
            for current_node_in_cone in sorted_cone_nodes_topo:
                if current_node_in_cone.name in node_values_in_cone: # Already processed as an input or something
                    continue
                
                if current_node_in_cone.const_val is not None:
                    node_values_in_cone[current_node_in_cone.name] = \
                        full_mask if current_node_in_cone.const_val == 1 else 0
                else:
                    output_mask = 0
                    if current_node_in_cone.patterns_one:
                        # For each pattern yielding '1', compute its mask and OR into output_mask
                        for p_mask_str in current_node_in_cone.patterns_one:
                            product_term_mask = full_mask
                            for idx_in, char_val in enumerate(p_mask_str):
                                fanin_node_name = current_node_in_cone.fanin_names[idx_in]
                                fanin_value_mask = node_values_in_cone[fanin_node_name] # Must be already computed
                                if char_val == '1':
                                    product_term_mask &= fanin_value_mask
                                elif char_val == '0':
                                    product_term_mask &= (~fanin_value_mask) & full_mask
                                # If char_val == '-', it's a don't care, doesn't modify product_term_mask
                            output_mask |= product_term_mask
                        node_values_in_cone[current_node_in_cone.name] = output_mask
                    elif current_node_in_cone.patterns_zero:
                        # For each pattern yielding '0', compute its mask and OR into zero_mask
                        zero_mask = 0
                        for p_mask_str in current_node_in_cone.patterns_zero:
                            product_term_mask = full_mask
                            for idx_in, char_val in enumerate(p_mask_str):
                                fanin_node_name = current_node_in_cone.fanin_names[idx_in]
                                fanin_value_mask = node_values_in_cone[fanin_node_name] # Must be already computed
                                if char_val == '1':
                                    product_term_mask &= fanin_value_mask
                                elif char_val == '0':
                                    product_term_mask &= (~fanin_value_mask) & full_mask
                            zero_mask |= product_term_mask
                        # The actual output is NOT (zero_mask)
                        node_values_in_cone[current_node_in_cone.name] = (~zero_mask) & full_mask
                    # If both patterns_one and patterns_zero are empty (no .names section after input/output names),
                    # it means the output is typically determined by other rules or is an error.
                    # Assuming problem structure implies one or the other will be populated for internal nodes.

            # The final truth table for the root `n_node` is `node_values_in_cone[n_node.name]`
            root_truth_table_mask = node_values_in_cone[n_node.name]

            # Write the .names block for this LUT
            f.write(f'.names {" ".join(inputs)} {n_node.name}\n')
            
            # Convert the bitmask truth table into BLIF patterns
            for j in range(num_patterns_N):
                if (root_truth_table_mask >> j) & 1: # If the j-th output bit is 1
                    # Reconstruct the input pattern for this j-th combination
                    input_pattern_str = ''.join('1' if (j >> i) & 1 else '0' for i in range(num_inputs_k))
                    f.write(f'{input_pattern_str} 1\n')

        f.write('.end\n')

def solve(input_file: str, solution_file: str):
    # K is hardcoded to 6 for this problem
    K = 6 
    solve_v2(input_file, solution_file, K)

if __name__ == "__main__":
    # Example usage:
    # python your_script.py input.blif output.blif
    if len(sys.argv) != 3:
        print("Usage: python your_script.py <input_blif_file> <output_blif_file>")
        sys.exit(1)
    
    input_blif = sys.argv[1]
    output_blif = sys.argv[2]
    solve(input_blif, output_blif)
