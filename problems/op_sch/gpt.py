import json
import collections

# Define the solve function as per the problem description
def solve(input_file: str, solution_file: str):
    class HLSchedulerSolver:
        def __init__(self, problem_data):
            self.problem_data = problem_data
            self.op_names_to_ids = {}
            self.op_ids_to_names = {}
            self.resource_names_to_ids = {}

            self.num_ops = 0
            self.op_delays = []
            self.op_resource_types = []

            self.num_resource_types = 0
            self.resource_limits = []

            # Adjacency lists for graph
            self.adj = [] # successors
            self.rev_adj = [] # predecessors
            self.in_degree = [] # incoming edges for dependency tracking
            self.out_degree = [] # outgoing edges for ALAP / sink detection

            self._parse_input()

        def _parse_input(self):
            nodes_data = self.problem_data["nodes"]
            self.num_ops = len(nodes_data)

            # Assign integer IDs to operations
            for i, (name, _) in enumerate(nodes_data):
                self.op_names_to_ids[name] = i
                self.op_ids_to_names[i] = name

            raw_delay_map = self.problem_data["delay"]
            raw_resource_limit_map = self.problem_data["resource"]

            # Assign integer IDs to resource types deterministically
            current_res_id = 0
            # Sort resource names to ensure consistent ID assignment across runs
            sorted_resource_names = sorted(list(raw_delay_map.keys()))

            for res_name in sorted_resource_names:
                self.resource_names_to_ids[res_name] = current_res_id
                # Ensure we populate resource_limits in the order of resource_ids
                self.resource_limits.append(raw_resource_limit_map[res_name])
                current_res_id += 1
            self.num_resource_types = current_res_id

            # Populate operation delays and resource types based on their IDs
            self.op_delays = [0] * self.num_ops
            self.op_resource_types = [-1] * self.num_ops
            for i, (node_name, res_type_name) in enumerate(nodes_data):
                op_id = self.op_names_to_ids[node_name]
                res_id = self.resource_names_to_ids[res_type_name]
                self.op_resource_types[op_id] = res_id
                self.op_delays[op_id] = raw_delay_map[res_type_name]

            # Build adjacency lists and degree arrays
            self.adj = [[] for _ in range(self.num_ops)]
            self.rev_adj = [[] for _ in range(self.num_ops)]
            self.in_degree = [0] * self.num_ops
            self.out_degree = [0] * self.num_ops

            for u_name, v_name, _ in self.problem_data["edges"]:
                u = self.op_names_to_ids[u_name]
                v = self.op_names_to_ids[v_name]
                self.adj[u].append(v)
                self.rev_adj[v].append(u)
                self.in_degree[v] += 1
                self.out_degree[u] += 1

        def _compute_asap_times(self):
            """
            Computes the As Soon As Possible (ASAP) start time for each operation,
            considering only data dependencies. This is essentially a longest path computation
            from source nodes.
            """
            asap = [0] * self.num_ops # Initialize with 0, as start time can be 0
            queue = collections.deque()
            current_in_degree = list(self.in_degree)

            # Add all source nodes (nodes with no predecessors) to the queue
            for i in range(self.num_ops):
                if current_in_degree[i] == 0:
                    queue.append(i)

            # Process nodes in topological order
            while queue:
                u = queue.popleft()
                for v_succ in self.adj[u]:
                    # ASAP[v_succ] must be at least the finish time of its latest predecessor 'u'
                    asap[v_succ] = max(asap[v_succ], asap[u] + self.op_delays[u])
                    current_in_degree[v_succ] -= 1
                    if current_in_degree[v_succ] == 0:
                        queue.append(v_succ)
            
            return asap

        def _compute_alap_times(self, target_latency, asap_times):
            """
            Computes the As Late As Possible (ALAP) start time for each operation,
            given a target latency, respecting data dependencies and ASAP times.
            Returns None if the target_latency is infeasible (i.e., ALAP < ASAP for any node).
            """
            alap = [target_latency] * self.num_ops # Initialize with target_latency (upper bound)
            queue = collections.deque()
            current_out_degree = list(self.out_degree)

            # Add all sink nodes (nodes with no successors) to the queue
            for i in range(self.num_ops):
                if current_out_degree[i] == 0: # Sink node
                    # ALAP start time for a sink node is such that it finishes by target_latency
                    alap[i] = target_latency - self.op_delays[i]
                    queue.append(i)
                
                # Preliminary check: ALAP must be >= ASAP (even before full ALAP propagation)
                # This catches cases where target_latency is too small for basic critical paths.
                if alap[i] < asap_times[i]:
                    return None # Target latency is too small, immediately infeasible

            # Process nodes in reverse topological order (from sinks to sources)
            while queue:
                u = queue.popleft()
                
                # Crucial step: Ensure ALAP doesn't go below ASAP for current node.
                # This also helps propagate stricter deadlines if a node's ALAP was
                # initially set too high but ASAP prevents it.
                alap[u] = max(alap[u], asap_times[u]) 
                if alap[u] < asap_times[u]:
                    return None # Target latency is infeasible due to ASAP conflict

                for v_pred in self.rev_adj[u]: # v_pred is a predecessor of u
                    # ALAP[v_pred] must be such that v_pred finishes before u starts
                    # So, v_pred must start at most (ALAP[u] - self.op_delays[v_pred])
                    alap[v_pred] = min(alap[v_pred], alap[u] - self.op_delays[v_pred])
                    current_out_degree[v_pred] -= 1
                    if current_out_degree[v_pred] == 0:
                        queue.append(v_pred)

            # Final pass to ensure ALAP >= ASAP for all nodes, considering full propagation.
            # This is a safeguard against any subtle inconsistencies after traversal.
            for i in range(self.num_ops):
                alap[i] = max(alap[i], asap_times[i])
                if alap[i] < asap_times[i]:
                    return None # Target latency is infeasible
            
            return alap

        def _compute_critical_path_lengths_to_sink(self):
            """
            Computes the length of the longest path from the start of each node
            to the end of any reachable sink node. This metric is valuable for
            prioritizing operations in list scheduling (criticality heuristic).
            """
            cp_lengths = [0] * self.num_ops 
            queue = collections.deque()
            current_out_degree = list(self.out_degree)

            # Add all sink nodes to the queue (their critical path length is just their own delay)
            for i in range(self.num_ops):
                if current_out_degree[i] == 0: # Sink node
                    queue.append(i)
                    cp_lengths[i] = self.op_delays[i]

            # Process nodes in reverse topological order (from sinks to sources)
            while queue:
                u = queue.popleft()
                for v_pred in self.rev_adj[u]: # v_pred is a predecessor of u
                    # Critical path length from v_pred is its own delay plus the longest
                    # critical path length from any of its successors.
                    path_len_via_u = self.op_delays[v_pred] + cp_lengths[u]
                    cp_lengths[v_pred] = max(cp_lengths[v_pred], path_len_via_u)
                    
                    current_out_degree[v_pred] -= 1
                    if current_out_degree[v_pred] == 0:
                        queue.append(v_pred)
            
            return cp_lengths
        
        def _priority_function(self, op_id, current_alap_times, critical_path_lengths_to_sink):
            """
            Defines the multi-criteria priority for list scheduling. Smaller values indicate higher priority.
            This heuristic aims to schedule urgent and critical operations first:
            1. Smallest ALAP time: Operations with less "mobility" (earliest deadline) are prioritized.
            2. Largest critical path length to a sink: Operations that lie on longer critical paths
               (and thus have a greater impact on overall latency) are prioritized.
            3. Smallest operation ID: Provides deterministic tie-breaking for consistency.
            """
            # Use negative for critical path length because Python's sort is ascending,
            # but we want to prioritize operations with *larger* critical path lengths.
            return (current_alap_times[op_id], -critical_path_lengths_to_sink[op_id], op_id)


        def _list_schedule(self, target_latency, asap_times_static, cp_lengths_static):
            """
            Attempts to find a feasible schedule for a given target_latency using
            a priority-based list scheduling approach.
            Returns the schedule (list of start times) if successful, None otherwise.
            """
            # If there are no operations, it's considered successfully scheduled with latency 0.
            if self.num_ops == 0:
                return [] 

            # Verify immediate feasibility based on pre-calculated ASAP times and target latency
            min_latency_from_asap_alone = 0
            for i in range(self.num_ops):
                min_latency_from_asap_alone = max(min_latency_from_asap_alone, asap_times_static[i] + self.op_delays[i])
            
            if target_latency < min_latency_from_asap_alone:
                return None # Target latency is too aggressive purely based on data dependencies

            # Compute ALAP times for the given target_latency. This calculation must be successful.
            alap_times = self._compute_alap_times(target_latency, asap_times_static)
            if alap_times is None:
                return None # Infeasible target_latency due to ALAP/ASAP conflicts propagated

            scheduled_start_times = [-1] * self.num_ops
            num_scheduled_ops = 0

            # `resource_usage[resource_type_id][time_step]` stores the number of units used.
            # Array size is `target_latency` because operations can occupy cycles from 0 up to `target_latency - 1`.
            # For example, if target_latency is 4, cycles are 0, 1, 2, 3. An op starting at 0 with delay 4 uses 0,1,2,3.
            resource_usage = [[0] * target_latency for _ in range(self.num_resource_types)]
            
            # `current_in_degree_for_ready_check` is a mutable copy to track completed predecessors
            current_in_degree_for_ready_check = list(self.in_degree)

            # `ready_ops_for_consideration`: Set of operations whose predecessors have all finished AND haven't been scheduled yet
            ready_ops_for_consideration = set() 
            for i in range(self.num_ops):
                if self.in_degree[i] == 0: # Source nodes start as ready
                    ready_ops_for_consideration.add(i)

            # `ops_in_progress`: Dictionary tracking operations currently executing: `op_id -> finish_time`
            ops_in_progress = {}

            current_time = 0
            # Continue scheduling as long as there are operations yet to be scheduled
            # and `current_time` hasn't exceeded the `target_latency` limit.
            while num_scheduled_ops < self.num_ops:

                # If `current_time` reaches `target_latency`, it means no more operations
                # can START if they have any positive delay, as they would finish beyond the target.
                if current_time >= target_latency: 
                    return None # Failed to schedule within target_latency

                # 1. Identify operations that are finishing at the start of `current_time`
                # (i.e., they started and finished in cycles ending before `current_time`).
                ops_just_finished_this_step = []
                for op_id_progress, finish_time in list(ops_in_progress.items()):
                    if finish_time == current_time:
                        ops_just_finished_this_step.append(op_id_progress)
                        del ops_in_progress[op_id_progress] # Remove from in-progress

                # 2. Update `ready_ops_for_consideration` based on operations that just finished
                for op_id_finished in ops_just_finished_this_step:
                    for successor_op_id in self.adj[op_id_finished]:
                        current_in_degree_for_ready_check[successor_op_id] -= 1
                        if current_in_degree_for_ready_check[successor_op_id] == 0:
                            ready_ops_for_consideration.add(successor_op_id)

                # 3. Select `candidate_ops_for_scheduling` from `ready_ops_for_consideration`
                # that also respect their ALAP times and can finish within `target_latency`.
                candidate_ops_for_scheduling = []
                for op_id in list(ready_ops_for_consideration): # Iterate over a copy for safe modification
                    # If current_time exceeds an operation's ALAP, this `target_latency` is infeasible.
                    if current_time > alap_times[op_id]:
                        return None 
                    
                    # An operation starting now must be able to finish by `target_latency`.
                    # An operation starting at `t_i` with `delay d_i` finishes *at* `t_i + d_i`.
                    # This finish time must be `<= target_latency`.
                    if current_time + self.op_delays[op_id] > target_latency:
                        # This operation cannot be scheduled at `current_time` or any later time
                        # within the given `target_latency`. This indicates infeasibility.
                        return None 
                    
                    candidate_ops_for_scheduling.append(op_id)

                # 4. Sort candidates by the defined priority function
                candidate_ops_for_scheduling.sort(
                    key=lambda op_id_lambda: self._priority_function(
                        op_id_lambda, alap_times, cp_lengths_static
                    )
                )

                # 5. Attempt to schedule operations based on priority and resource availability
                for op_id_to_schedule in candidate_ops_for_scheduling:
                    res_type = self.op_resource_types[op_id_to_schedule]
                    delay = self.op_delays[op_id_to_schedule]

                    # Check resource availability for the duration [current_time, current_time + delay - 1]
                    can_schedule_op = True
                    for t_step in range(current_time, current_time + delay):
                        # Ensure we don't try to use resources beyond the `target_latency` boundary.
                        # This check is largely a safeguard given the `current_time + delay > target_latency` check above.
                        if t_step >= target_latency: 
                            can_schedule_op = False
                            break
                        if resource_usage[res_type][t_step] >= self.resource_limits[res_type]:
                            can_schedule_op = False
                            break
                    
                    if can_schedule_op:
                        scheduled_start_times[op_id_to_schedule] = current_time
                        num_scheduled_ops += 1
                        
                        # Remove from ready set as it's now scheduled
                        ready_ops_for_consideration.remove(op_id_to_schedule)
                        # Mark as in progress until its finish time
                        ops_in_progress[op_id_to_schedule] = current_time + delay

                        # Update resource usage for each cycle the operation is active
                        for t_step in range(current_time, current_time + delay):
                            resource_usage[res_type][t_step] += 1
                
                # If no operations are in progress, no new operations are ready,
                # but there are still operations that need to be scheduled,
                # it implies a deadlock or an unresolvable state.
                if not ops_in_progress and not ready_ops_for_consideration and num_scheduled_ops < self.num_ops:
                    return None 

                # Move to the next time step
                current_time += 1

            # If the loop completes, it means all operations have been successfully scheduled within the `target_latency`.
            return scheduled_start_times

        def run_solver(self):
            # Handle empty graph explicitly, no operations means latency 0.
            if self.num_ops == 0:
                return {}

            # Pre-calculate static graph properties that don't depend on the specific target latency.
            asap_times_static = self._compute_asap_times()
            cp_lengths_static = self._compute_critical_path_lengths_to_sink()
            
            # Determine the binary search range for the minimum latency.
            # Lower bound (`low`): The minimum possible latency, dictated by the longest
            # dependency chain (critical path), assuming infinite resources.
            low = 0
            for i in range(self.num_ops):
                low = max(low, asap_times_static[i] + self.op_delays[i])

            # Upper bound (`high`): A loose but safe upper bound. Sum of all operation delays
            # ensures that even if operations were scheduled one after another (max possible latency),
            # it would still be covered.
            high = sum(self.op_delays) if self.num_ops > 0 else 0
            # Ensure `high` is at least `low` for the binary search range to be valid.
            high = max(low, high) 

            best_schedule_array = None
            
            # Binary search for the minimum latency
            while low <= high:
                mid_latency_target = low + (high - low) // 2
                
                # Attempt to find a feasible schedule for the current `mid_latency_target`.
                schedule_attempt = self._list_schedule(
                    mid_latency_target, asap_times_static, cp_lengths_static
                )

                if schedule_attempt is not None:
                    # If a schedule was found for `mid_latency_target`, it's achievable.
                    # Store this schedule (it's the best so far for this target or smaller).
                    # Then, try to achieve an even smaller latency by searching the lower half.
                    best_schedule_array = schedule_attempt
                    high = mid_latency_target - 1
                else:
                    # If no schedule was found, `mid_latency_target` is too small.
                    # Increase the target latency and search in the upper half.
                    low = mid_latency_target + 1

            # Format the best found schedule into the required output dictionary.
            output_dict = {}
            if best_schedule_array is not None:
                for i in range(self.num_ops):
                    op_name = self.op_ids_to_names[i]
                    output_dict[op_name] = best_schedule_array[i]

            return output_dict

    # Main execution logic for the solve function
    with open(input_file, "r") as f:
        problem_data_json = json.load(f)

    # Create an instance of the HLSchedulerSolver and run it.
    solver_instance = HLSchedulerSolver(problem_data_json)
    solution_dict = solver_instance.run_solver()

    # Write the solution to the output file in the specified format.
    with open(solution_file, "w") as f:
        if solution_dict: # Only write if a valid schedule was found (not empty dict)
            output_lines = []
            # Output nodes in the order they were provided in the input "nodes" list for consistency.
            for node_name_in_order, _ in problem_data_json["nodes"]:
                if node_name_in_order in solution_dict:
                    output_lines.append(
                        f"{node_name_in_order}:{solution_dict[node_name_in_order]}"
                    )
            f.write("\n".join(output_lines))
