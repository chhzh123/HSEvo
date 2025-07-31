import csv
from datetime import datetime, timedelta
import collections
import heapq
from scipy.optimize import linprog

# Constants
BASE = "NKX"
C_POS = 10000.0
H_D_MAX_MIN = 14 * 60
H_B_MAX_MIN = 10 * 60
L_MAX = 6
R_MIN_MIN = 9 * 60

# Use a large number for infinity
INF = float('inf')

class Leg:
    def __init__(self, flt_num, dptr_date, dptr_time, dptr_stn, arrv_date, arrv_time, arrv_stn, comp, duty_cost, paring_cost, index):
        self.flt_num = flt_num
        self.dptr_date_str = dptr_date
        self.dptr_time_str = dptr_time
        self.dptr_stn = dptr_stn
        self.arrv_date_str = arrv_date
        self.arrv_time_str = arrv_time
        self.arrv_stn = arrv_stn
        self.comp = comp
        self.duty_cost_per_hour = duty_cost
        self.paring_cost_per_hour = paring_cost
        self.index = index

        self.dptr_dt = datetime.strptime(f"{dptr_date} {dptr_time}", "%m/%d/%Y %H:%M")
        self.arrv_dt = datetime.strptime(f"{arrv_date} {arrv_time}", "%m/%d/%Y %H:%M")
        self.block_minutes = int((self.arrv_dt - self.dptr_dt).total_seconds() / 60)

    def get_token(self):
        return f"{self.flt_num}_{self.dptr_dt.strftime('%Y-%m-%d')}"

def parse_input(input_file):
    legs = []
    duty_cost = None
    paring_cost = None

    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Forward fill costs
            if row['DutyCostPerHour']:
                duty_cost = float(row['DutyCostPerHour'])
            if row['ParingCostPerHour']:
                paring_cost = float(row['ParingCostPerHour'])

            leg = Leg(
                flt_num=row['FltNum'],
                dptr_date=row['DptrDate'],
                dptr_time=row['DptrTime'],
                dptr_stn=row['DptrStn'],
                arrv_date=row['ArrvDate'],
                arrv_time=row['ArrvTime'],
                arrv_stn=row['ArrvStn'],
                comp=row['Comp'],
                duty_cost=duty_cost,
                paring_cost=paring_cost,
                index=i
            )
            legs.append(leg)

    # Sort legs by departure time
    legs.sort(key=lambda x: x.dptr_dt)

    # Update indices after sorting
    for i, leg in enumerate(legs):
        leg.index = i

    return legs

def calculate_pairing_cost(pairing_legs_indices, legs_data):
    """Calculates the total cost of a feasible pairing given leg indices."""
    total_duty_cost = 0
    total_block_cost = 0
    positioning_cost = 0

    if not pairing_legs_indices:
        return 0

    # Check positioning cost
    first_leg_idx = pairing_legs_indices[0]
    if legs_data[first_leg_idx].dptr_stn != BASE:
        positioning_cost = C_POS

    current_duty_start_leg_idx = pairing_legs_indices[0]

    for i, leg_idx in enumerate(pairing_legs_indices):
        leg = legs_data[leg_idx]
        total_block_cost += leg.block_minutes * leg.paring_cost_per_hour / 60.0 # Convert minutes to hours

        is_last_leg_in_pairing = (i == len(pairing_legs_indices) - 1)
        requires_rest_for_next = False
        if not is_last_leg_in_pairing:
            next_leg_idx = pairing_legs_indices[i+1]
            next_leg = legs_data[next_leg_idx]
            rest_time_minutes = int((next_leg.dptr_dt - leg.arrv_dt).total_seconds() / 60)
            if rest_time_minutes >= R_MIN_MIN:
                 requires_rest_for_next = True

        if is_last_leg_in_pairing or requires_rest_for_next:
            # Calculate duty cost for the completed duty
            first_leg_in_duty = legs_data[current_duty_start_leg_idx]
            last_leg_in_duty = legs_data[leg_idx]
            duty_minutes = int((last_leg_in_duty.arrv_dt - first_leg_in_duty.dptr_dt).total_seconds() / 60)
            total_duty_cost += duty_minutes * first_leg_in_duty.duty_cost_per_hour / 60.0 # Convert minutes to hours

            # Start a new duty for the next leg if it exists
            if not is_last_leg_in_pairing:
                 current_duty_start_leg_idx = pairing_legs_indices[i+1]


    total_cost = total_duty_cost + total_block_cost + positioning_cost
    return total_cost


def solve_pricing_problem(legs, duals):
    """
    Solves the pricing problem using Dijkstra to find a minimum reduced cost pairing.
    State: (current_leg_idx, first_leg_in_duty_idx, duty_block_minutes, duty_leg_count)
    """
    num_legs = len(legs)
    # State: (current_leg_idx, first_leg_in_duty_idx, duty_block_minutes, duty_leg_count)
    dist = {} # {state: reduced_cost}
    pred = {} # {state: previous_state}
    pq = [] # (reduced_cost, current_leg_idx, first_leg_in_duty_idx, duty_block_minutes, duty_leg_count)

    # Virtual start state: (-1, -1, 0, 0)
    # Initial transitions from virtual start node to any leg k as the first leg of a pairing
    for k in range(num_legs):
        leg_k = legs[k]

        # Check if leg k can form a legal single-leg duty (as it's the first and only leg in the initial duty)
        if leg_k.block_minutes > H_B_MAX_MIN or 1 > L_MAX or leg_k.block_minutes > H_D_MAX_MIN:
             continue # Cannot start a pairing with this leg if it violates single-duty rules

        state_k = (k, k, leg_k.block_minutes, 1) # (current_leg_idx, first_leg_in_duty_idx, ...)

        # Reduced cost for starting with leg k
        # Cost = Block cost + Positioning cost - pi_k
        initial_reduced_cost = (leg_k.block_minutes / 60.0) * leg_k.paring_cost_per_hour - duals[k]
        if leg_k.dptr_stn != BASE:
            initial_reduced_cost += C_POS

        dist[state_k] = initial_reduced_cost
        pred[state_k] = (-1, -1, 0, 0) # Indicate start
        heapq.heappush(pq, (initial_reduced_cost, k, k, leg_k.block_minutes, 1))

    min_reduced_cost_pairing = INF
    best_pairing_end_state = None # (current_leg_idx, first_leg_in_duty_idx, duty_block_minutes, duty_leg_count)

    # Dijkstra
    while pq:
        cost, i, j, b, l = heapq.heappop(pq)
        current_state = (i, j, b, l)

        if cost > dist.get(current_state, INF):
            continue

        leg_i = legs[i]
        leg_j = legs[j] # First leg of current duty

        # Consider extending with any leg k that departs from leg_i.arrv_stn after leg_i.arrv_dt
        # Iterate through all legs k > i (chronological order)
        for k in range(i + 1, num_legs):
            leg_k = legs[k]

            # Check if leg k departs from the arrival station of leg i
            if leg_k.dptr_stn != leg_i.arrv_stn:
                continue

            # Check chronological order (already guaranteed by k > i and sorted legs, but double check time)
            if leg_k.dptr_dt < leg_i.arrv_dt:
                 continue

            # Option 1: Start a new duty with leg k
            rest_time_minutes = int((leg_k.dptr_dt - leg_i.arrv_dt).total_seconds() / 60)
            if rest_time_minutes >= R_MIN_MIN:
                new_l = 1
                new_b = leg_k.block_minutes
                new_j = k # New duty starts with leg k
                new_state = (k, new_j, new_b, new_l)

                # Check legality of the new single-leg duty (leg k)
                if new_b > H_B_MAX_MIN or new_l > L_MAX or new_b > H_D_MAX_MIN:
                    continue # Cannot start a new duty with this leg if it violates single-duty rules

                # Cost of previous duty (the one ending at leg i)
                prev_duty_start_leg = legs[j]
                prev_duty_minutes = int((leg_i.arrv_dt - prev_duty_start_leg.dptr_dt).total_seconds() / 60)
                cost_prev_duty = (prev_duty_minutes / 60.0) * prev_duty_start_leg.duty_cost_per_hour

                # Edge cost = Cost of previous duty + Block cost of leg k - pi_k
                edge_cost = cost_prev_duty + (leg_k.block_minutes / 60.0) * leg_k.paring_cost_per_hour - duals[k]
                new_total_cost = cost + edge_cost

                if new_total_cost < dist.get(new_state, INF):
                    dist[new_state] = new_total_cost
                    pred[new_state] = current_state
                    heapq.heappush(pq, (new_total_cost, k, new_j, new_b, new_l))

            # Option 2: Continue current duty with leg k
            new_l = l + 1
            new_b = b + leg_k.block_minutes
            new_j = j # Current duty started with the same leg j
            new_state = (k, new_j, new_b, new_l)

            # Check legality constraints for continuing duty
            duty_start_leg = legs[j]
            new_duty_minutes = int((leg_k.arrv_dt - duty_start_leg.dptr_dt).total_seconds() / 60)

            if (new_duty_minutes <= H_D_MAX_MIN and
                new_b <= H_B_MAX_MIN and
                new_l <= L_MAX):

                # Edge cost = Block cost of leg k - pi_k
                edge_cost = (leg_k.block_minutes / 60.0) * leg_k.paring_cost_per_hour - duals[k]
                new_total_cost = cost + edge_cost

                if new_total_cost < dist.get(new_state, INF):
                    dist[new_state] = new_total_cost
                    pred[new_state] = current_state
                    heapq.heappush(pq, (new_total_cost, k, new_j, new_b, new_l))

    # After Dijkstra, find the minimum reduced cost pairing among all reachable end states
    min_reduced_cost_pairing = INF
    best_pairing_end_state = None

    for state, cost in dist.items():
        i, j, b, l = state
        leg_i = legs[i]
        leg_j = legs[j] # First leg of current duty

        # Reduced cost of the full pairing ending in this state
        # = Path cost (cost) + Cost of the current duty
        current_duty_minutes = int((leg_i.arrv_dt - leg_j.dptr_dt).total_seconds() / 60)
        cost_current_duty = (current_duty_minutes / 60.0) * leg_j.duty_cost_per_hour

        pairing_reduced_cost = cost + cost_current_duty

        if pairing_reduced_cost < min_reduced_cost_pairing:
            min_reduced_cost_pairing = pairing_reduced_cost
            best_pairing_end_state = state

    # Reconstruct the pairing if a negative reduced cost pairing was found
    found_pairing_indices = None
    if best_pairing_end_state is not None and min_reduced_cost_pairing < -1e-6: # Use a small tolerance
        found_pairing_indices = []
        current_state = best_pairing_end_state
        while current_state != (-1, -1, 0, 0): # Stop when we reach the virtual start node
            leg_idx, _, _, _ = current_state
            found_pairing_indices.append(leg_idx)
            current_state = pred[current_state]
        found_pairing_indices.reverse() # The path was built backwards

    return found_pairing_indices, min_reduced_cost_pairing


def solve(input_file: str, solution_file: str):
    legs = parse_input(input_file)
    num_legs = len(legs)

    if num_legs == 0:
        with open(solution_file, 'w') as f:
            pass
        return

    # Column Generation
    pairings = [] # List of lists of leg indices
    pairing_costs = [] # List of costs for each pairing

    # Artificial variables setup
    M = 1e6 # High cost for artificial variables
    # Initial RMP has only artificial variables
    current_c = [M] * num_legs
    current_A_eq = [[0.0] * num_legs for _ in range(num_legs)]
    for i in range(num_legs):
        current_A_eq[i][i] = 1.0 # Identity matrix for artificials
    current_b_eq = [1.0] * num_legs
    current_bounds = [(0.0, None)] * num_legs # x_p >= 0, a_f >= 0

    # Keep track of which columns are artificial
    is_artificial = [True] * num_legs

    iteration = 0
    max_iterations = 100 # Limit iterations to prevent excessive runtime

    while iteration < max_iterations:
        iteration += 1

        # Solve LP relaxation of RMP
        # Use highs solver
        res = linprog(current_c, A_eq=current_A_eq, b_eq=current_b_eq, bounds=current_bounds, method='highs')

        if not res.success:
            # If LP is infeasible, the original problem is likely infeasible.
            if res.status == 2: # Infeasible
                 with open(solution_file, 'w') as f:
                     pass
                 return
            # If LP fails for other reasons, break.
            break

        # Get dual values
        duals = res.v # Duals for the equality constraints (one per leg)

        # Solve Pricing Problem
        new_pairing_indices, min_reduced_cost = solve_pricing_problem(legs, duals)

        # Check if a profitable column was found
        if new_pairing_indices and min_reduced_cost < -1e-6: # Use tolerance
            # Calculate the actual cost of the new pairing
            new_pairing_cost = calculate_pairing_cost(new_pairing_indices, legs)

            # Add the new pairing column to the RMP
            pairings.append(new_pairing_indices)
            pairing_costs.append(new_pairing_cost)

            # Update A_eq: add a new column for the new pairing
            new_column = [0.0] * num_legs
            for leg_idx in new_pairing_indices:
                new_column[leg_idx] = 1.0

            # Insert the new column before the artificial columns
            for row in current_A_eq:
                row.insert(len(pairings) - 1, 0.0) # Insert 0.0 initially

            for i in range(num_legs):
                 current_A_eq[i][len(pairings) - 1] = new_column[i]

            # Update c and bounds
            current_c.insert(len(pairings) - 1, new_pairing_cost)
            current_bounds.insert(len(pairings) - 1, (0.0, None))
            is_artificial.insert(len(pairings) - 1, False)

        else:
            # No profitable column found, LP optimum reached
            break # Exit column generation loop

    # Remove artificial columns before solving IP
    final_pairings = []
    final_pairing_costs = []

    # Filter out artificial columns
    col_indices_to_keep = [i for i, is_art in enumerate(is_artificial) if not is_art]

    if not col_indices_to_keep:
        # No non-artificial columns generated. Problem likely infeasible.
        with open(solution_file, 'w') as f:
            pass
        return

    final_pairings = [pairings[i] for i in col_indices_to_keep]
    final_pairing_costs = [pairing_costs[i] for i in col_indices_to_keep]

    # Build the final A_eq matrix using only non-artificial columns
    final_A_eq = [[0.0] * len(final_pairings) for _ in range(num_legs)]
    for p_idx, pairing_legs_indices in enumerate(final_pairings):
        for leg_idx in pairing_legs_indices:
            final_A_eq[leg_idx][p_idx] = 1.0

    final_b_eq = [1.0] * num_legs
    final_bounds = [(0.0, 1.0)] * len(final_pairings) # x_p in {0, 1}

    # Solve the final IP
    res_ip = linprog(final_pairing_costs, A_eq=final_A_eq, b_eq=final_b_eq, bounds=final_bounds, method='highs', integrality=1)

    if not res_ip.success:
        # IP did not converge or is infeasible.
        with open(solution_file, 'w') as f:
            pass
        return

    # Extract selected pairings from IP solution
    selected_pairings_indices_in_final_list = [i for i, val in enumerate(res_ip.x) if val > 0.5] # Use tolerance for binary variables

    # Write output file
    with open(solution_file, 'w') as f:
        for p_idx in selected_pairings_indices_in_final_list:
            pairing_legs_indices = final_pairings[p_idx]
            # Ensure legs are in chronological order within the pairing for output
            # They should already be chronological from the pricing problem construction
            # Sort is redundant if pricing problem builds chronological paths, but safe.
            pairing_legs_indices.sort(key=lambda idx: legs[idx].dptr_dt)
            tokens = [legs[idx].get_token() for idx in pairing_legs_indices]
            f.write(" ".join(tokens) + "\n")
