def solve(input_file: str, solution_file: str):
    # Helper functions needed inside solve
    
    from dataclasses import dataclass
    from typing import Dict, List
    import os
    import random
    import math
    import time
    import numpy
    import networkx
    import pandas

    @dataclass
    class Node:
        index: int
        x: float
        y: float
        demand: float
        earliest_time: float
        latest_time: float
        service_time: float
        pickup_sibling: int
        delivery_sibling: int
        is_pickup: bool


    @dataclass
    class Instance:
        name: str
        type: str
        dimension: int
        vehicles: int
        capacity: float
        edge_weight_type: str
        nodes: Dict[int, Node]
        depot_node: List[int]


    def read_instance(file_path: str) -> Instance:
        with open(file_path, "r") as file:
            lines = file.readlines()

        name = ""
        type_ = ""
        dimension = 0
        vehicles = 0
        capacity = 0.0
        edge_weight_type = ""
        nodes = {}
        depot_node = []

        section = None
        for line in lines:
            line = line.strip()
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("TYPE"):
                type_ = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("VEHICLES"):
                vehicles = int(line.split(":")[1].strip())
            elif line.startswith("CAPACITY"):
                capacity = float(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                section = "NODE_COORD_SECTION"
            elif line.startswith("PICKUP_AND_DELIVERY_SECTION"):
                section = "PICKUP_AND_DELIVERY_SECTION"
            elif line.startswith("DEPOT_SECTION"):
                section = "DEPOT_SECTION"
            elif line.startswith("EOF"):
                break
            elif section == "NODE_COORD_SECTION":
                parts = line.split()
                index = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                nodes[index] = Node(
                    index=index,
                    x=x,
                    y=y,
                    demand=0.0,
                    earliest_time=0.0,
                    latest_time=0.0,
                    service_time=0.0,
                    pickup_sibling=0,
                    delivery_sibling=0,
                    is_pickup=False,
                )
            elif section == "PICKUP_AND_DELIVERY_SECTION":
                parts = line.split()
                index = int(parts[0])
                if index in nodes:
                    nodes[index].demand = float(parts[1])
                    nodes[index].earliest_time = float(parts[2])
                    nodes[index].latest_time = float(parts[3])
                    nodes[index].service_time = float(parts[4])
                    nodes[index].pickup_sibling = int(parts[5])
                    nodes[index].delivery_sibling = int(parts[6])
                    nodes[index].is_pickup = (nodes[index].demand > 0)

            elif section == "DEPOT_SECTION":
                if line != "-1":
                    depot_node.append(int(line))

        return Instance(
            name=name,
            type=type_,
            dimension=dimension,
            vehicles=vehicles,
            capacity=capacity,
            edge_weight_type=edge_weight_type,
            nodes=nodes,
            depot_node=depot_node,
        )


    def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return round(((x2 - x1) ** 2 + (y1 - y2) ** 2) ** 0.5, 0)

    def calculate_route_cost_and_feasibility(route_indices, nodes, capacity, depot_index):
        """
        Calculates the total distance and checks feasibility for a given route.
        Returns (total_distance, is_feasible)
        """
        if not route_indices or route_indices[0] != depot_index or route_indices[-1] != depot_index:
            return float('inf'), False

        if len(route_indices) == 2: # Depot -> Depot route
            return 0.0, True

        total_distance = 0
        current_time = 0 # Time at the start depot
        current_load = 0
        visited_pickups = set() # Indices of pickup nodes visited so far in this route

        prev_node_index = route_indices[0] # Depot
        prev_node = nodes[prev_node_index]
        departure_time_prev = 0 # Assume vehicle leaves depot at time 0

        is_feasible = True

        # Iterate through nodes in the route, starting from the first node after the initial depot
        for i in range(1, len(route_indices)):
            curr_node_index = route_indices[i]
            curr_node = nodes[curr_node_index]

            # Travel from prev_node to curr_node
            travel_time = calculate_distance(prev_node.x, prev_node.y, curr_node.x, curr_node.y)
            total_distance += travel_time

            # Arrival time at curr_node
            arrival_time = departure_time_prev + travel_time

            # Check time window at curr_node
            if arrival_time > curr_node.latest_time + 1e-9: # Add tolerance for float comparison
                is_feasible = False
                break

            # Calculate departure time from curr_node
            wait_time = max(0.0, curr_node.earliest_time - arrival_time)
            departure_time = arrival_time + wait_time + curr_node.service_time

            # Update load and check capacity at curr_node
            current_load += curr_node.demand

            if current_load > capacity + 1e-9: # Add tolerance for float comparison
                is_feasible = False
                break
            # Load must be non-negative for standard PDPTW
            if current_load < -1e-9: # Use a small tolerance for float comparison
                 is_feasible = False
                 break

            # Check pick-up before delivery constraint (only for non-depot nodes)
            if curr_node_index != depot_index:
                if curr_node.is_pickup:
                    visited_pickups.add(curr_node.index)
                elif curr_node.demand < 0: # It's a delivery node
                    pickup_sibling_index = curr_node.pickup_sibling
                    # Check if the corresponding pickup has been visited *before* this delivery node in the route
                    if pickup_sibling_index not in visited_pickups:
                        is_feasible = False
                        break

            # Move to the next node
            prev_node = curr_node
            departure_time_prev = departure_time

        if not is_feasible:
            return float('inf'), False

        return total_distance, True

    def calculate_total_cost(routes, nodes, capacity, depot_index):
        total_cost = 0
        for route in routes:
            cost, feasible = calculate_route_cost_and_feasibility(route, nodes, capacity, depot_index)
            if not feasible:
                return float('inf')
            total_cost += cost
        return total_cost

    def find_best_pair_insertion_limited_vehicles(routes, p_index, d_index, instance, nodes, capacity, depot_index, max_vehicles):
         """
         Finds the best feasible insertion position for pair (p_index, d_index)
         into the current set of routes or a new route, respecting vehicle limit.
         Returns (best_route_idx, best_pos_p, best_pos_d, best_cost_increase)
         best_route_idx = -1 if no feasible insertion found.
         best_route_idx = len(routes) if new route is best.
         """
         best_cost_increase = float('inf')
         best_route_idx = -1
         best_pos_p = -1
         best_pos_d = -1

         p_node = nodes[p_index]
         d_node = nodes[d_index]

         # Try inserting into existing routes
         for route_idx, route in enumerate(routes):
             current_route_cost, current_route_feasible = calculate_route_cost_and_feasibility(route, nodes, capacity, depot_index)
             if not current_route_feasible: continue # Skip inserting into an already infeasible route

             # Iterate through all possible insertion positions for p (after first depot, before last depot)
             # Positions are indices in the list `route`.
             # Index 1 is after the first depot. Index len(route) - 1 is before the last depot.
             # If route is [1, 5, 6, 1], indices are 0, 1, 2, 3. Possible pos_p are 1, 2, 3.
             for pos_p in range(1, len(route)):
                 # Iterate through all possible insertion positions for d (after pos_p, before last depot)
                 # pos_d must be >= pos_p.
                 # Possible positions for d are from index `pos_p` to len(route) (inclusive).
                 # Index len(route) means before the final depot.
                 for pos_d in range(pos_p, len(route) + 1):

                     # Create the potential new route by inserting p and d
                     temp_route = route[:]
                     temp_route.insert(pos_p, p_index)
                     # After inserting p at pos_p, the index pos_d shifts by 1 if pos_d >= pos_p.
                     # Since we iterate pos_d >= pos_p, the index always shifts by 1.
                     adjusted_pos_d = pos_d + 1
                     temp_route.insert(adjusted_pos_d, d_index)
                     new_route_indices = temp_route

                     # Check feasibility and calculate cost of the new route
                     # This is the O(N) part
                     new_route_cost, new_route_feasible = calculate_route_cost_and_feasibility(new_route_indices, nodes, capacity, depot_index)

                     if new_route_feasible:
                         cost_increase = new_route_cost - current_route_cost
                         if cost_increase < best_cost_increase:
                             best_cost_increase = cost_increase
                             best_route_idx = route_idx
                             best_pos_p = pos_p
                             best_pos_d = pos_d

         # Try creating a new route for the pair, ONLY if we haven't reached the vehicle limit
         # Count routes that are currently *not* just [depot, depot]
         active_routes_count = len([r for r in routes if len(r) > 2])

         if active_routes_count < max_vehicles:
             new_route_indices = [depot_index, p_index, d_index, depot_index]
             new_route_cost, new_route_feasible = calculate_route_cost_and_feasibility(new_route_indices, nodes, capacity, depot_index)

             if new_route_feasible:
                 # Cost increase is just the cost of the new route, as the pair was unassigned (cost 0)
                 cost_increase = new_route_cost
                 if cost_increase < best_cost_increase:
                     best_cost_increase = cost_increase
                     # Indicate a new route. We will add it to the end of the current list of routes.
                     best_route_idx = len(routes)
                     best_pos_p = 1 # Position of p in [depot, p, d, depot]
                     best_pos_d = 2 # Position of d in [depot, p, d, depot]

         if best_route_idx != -1:
              return best_route_idx, best_pos_p, best_pos_d, best_cost_increase
         else:
              return -1, -1, -1, float('inf') # No feasible insertion found

    def apply_pair_insertion(routes, route_idx, pos_p, pos_d, p_index, d_index, depot_index):
        """Applies the insertion of pair (p_index, d_index) into the specified route."""
        if route_idx == len(routes): # Create a new route
            routes.append([depot_index, p_index, d_index, depot_index])
        else: # Insert into existing route
            route = routes[route_idx]
            route.insert(pos_p, p_index)
            # Adjust pos_d after inserting p
            adjusted_pos_d = pos_d + 1
            route.insert(adjusted_pos_d, d_index)

    # --- Main Solver Logic ---

    # Read the instance
    instance = read_instance(input_file)
    nodes = instance.nodes
    depot_index = instance.depot_node[0]
    capacity = instance.capacity
    num_vehicles = instance.vehicles

    # Get all pick-up node indices
    pickup_indices = [node.index for node in nodes.values() if node.is_pickup]
    # Create pairs (pickup_index, delivery_index)
    all_pairs = [(p_idx, nodes[p_idx].delivery_sibling) for p_idx in pickup_indices]

    # --- Initial Solution: Greedy Insertion ---
    start_time_init = time.time()
    current_routes = [] # Start with no routes
    unassigned_pairs = list(all_pairs)

    # Sort pairs by the earliest time of the pickup node
    unassigned_pairs.sort(key=lambda pair: nodes[pair[0]].earliest_time)

    while unassigned_pairs:
        # Check time limit during initial construction
        if time.time() - start_time_init > 0.8 * 98: # Use a fraction of total time limit
             break

        p_index, d_index = unassigned_pairs[0] # Take the first (earliest time) unassigned pair

        # Find the best place to insert this pair
        best_route_idx, best_pos_p, best_pos_d, best_cost_increase = find_best_pair_insertion_limited_vehicles(
            current_routes, p_index, d_index, instance, nodes, capacity, depot_index, num_vehicles
        )

        if best_route_idx == -1:
            # If initial construction fails to insert a pair, it's a problem.
            # Let's add the pair back to unassigned and break the loop.
            unassigned_pairs.append((p_index, d_index)) # Put it back
            break # Stop initial construction

        # Apply the insertion
        apply_pair_insertion(current_routes, best_route_idx, best_pos_p, best_pos_d, p_index, d_index, depot_index)

        # Remove the assigned pair
        unassigned_pairs.pop(0)

    initial_routes = [route for route in current_routes if len(route) > 2]
    initial_cost = calculate_total_cost(initial_routes, nodes, capacity, depot_index)

    # --- Large Neighborhood Search (LNS) ---
    start_time_lns = time.time()
    time_limit = 98 # seconds (leave a little time for writing output)

    current_solution = list(initial_routes)
    current_cost = calculate_total_cost(current_solution, nodes, capacity, depot_index)
    best_solution = list(current_solution)
    best_cost = current_cost

    min_requests_to_remove = 2
    fraction_to_remove = 0.15
    max_iterations = 5000 # High limit, rely on time

    iteration = 0
    while time.time() - start_time_lns < time_limit and iteration < max_iterations:
        iteration += 1

        # 1. Destroy Step: Randomly remove requests
        current_assigned_pairs = []
        nodes_in_solution = set()
        for route in current_solution:
            for node_idx in route:
                 if node_idx != depot_index:
                      nodes_in_solution.add(node_idx)

        current_assigned_pairs = []
        for p_idx, d_idx in all_pairs:
             if p_idx in nodes_in_solution and d_idx in nodes_in_solution:
                  current_assigned_pairs.append((p_idx, d_idx))

        if not current_assigned_pairs:
             break

        num_to_remove_this_iter = max(min_requests_to_remove, int(len(current_assigned_pairs) * fraction_to_remove))
        num_to_remove_this_iter = min(num_to_remove_this_iter, len(current_assigned_pairs))

        if num_to_remove_this_iter == 0:
             break

        pairs_to_remove = random.sample(current_assigned_pairs, num_to_remove_this_iter)
        removed_p_indices = {p for p, d in pairs_to_remove}
        removed_d_indices = {d for p, d in pairs_to_remove}
        removed_node_indices = removed_p_indices.union(removed_d_indices)

        temp_routes = []
        for route in current_solution:
            new_route = [node_idx for node_idx in route if node_idx not in removed_node_indices]
            if not new_route:
                 temp_routes.append([depot_index, depot_index])
            else:
                 if new_route[0] != depot_index: new_route.insert(0, depot_index)
                 if new_route[-1] != depot_index: new_route.append(depot_index)
                 cleaned_route = []
                 for i in range(len(new_route)):
                      if i > 0 and new_route[i] == depot_index and new_route[i-1] == depot_index:
                           continue
                      cleaned_route.append(new_route[i])
                 temp_routes.append(cleaned_route)

        current_routes_after_removal = temp_routes

        # 2. Repair Step: Reinsert removed pairs
        pairs_to_reinsert = list(pairs_to_remove)
        random.shuffle(pairs_to_reinsert)

        reinserted_successfully = True
        routes_during_repair = list(current_routes_after_removal)

        for p_index, d_index in pairs_to_reinsert:
            best_route_idx, best_pos_p, best_pos_d, best_cost_increase = find_best_pair_insertion_limited_vehicles(
                routes_during_repair, p_index, d_index, instance, nodes, capacity, depot_index, num_vehicles
            )

            if best_route_idx == -1:
                reinserted_successfully = False
                break

            apply_pair_insertion(routes_during_repair, best_route_idx, best_pos_p, best_pos_d, p_index, d_index, depot_index)

        # If repair was successful
        if reinserted_successfully:
            new_solution = [route for route in routes_during_repair if len(route) > 2]
            new_cost = calculate_total_cost(new_solution, nodes, capacity, depot_index)

            # 3. Acceptance Criterion: Simple Hill Climbing
            if new_cost < best_cost:
                best_cost = new_cost
                best_solution = list(new_solution)
                current_solution = list(new_solution)

    routes = best_solution

    # Write the solution to the output file
    with open(solution_file, 'w') as f:
        f.write(f"{instance.name}\n")
        for route in routes:
            f.write(" ".join(map(str, route)) + "\n")