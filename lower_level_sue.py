
import math

def solve_sue_msa(demand, stations, theta, tau, alpha, tax_list, convergence_tolerance=1e-3, max_iter=1000, step_cap=0.2):
    """
    Solves the Stochastic User Equilibrium (SUE) problem using the MSA algorithm.
    Inputs:
        demand (float): Total number of travelers (N).
        stations (dict): A dictionary of stations with their properties (p, c, T0).
        theta (float): Logit scale parameter.
        tau (float): Travel time cost coefficient.
        alpha (float): Congestion sensitivity coefficient.
        convergence_tolerance (float): Stopping criterion for MSA.
        max_iter (int): Maximum number of iterations.
    Outputs:
        dict: A dictionary of equilibrium flows for each station.
    """
    station_ids = list(stations.keys())
    flows = {j: demand / len(station_ids) for j in station_ids}  # Initialize flows equally

    for s in range(1, max_iter + 1):
        # Step 1.1: Calculate disutility (generalized cost) for current flows
        generalized_costs = {}
        for j in station_ids:
            p_j = stations[j]['price']
            c_j = stations[j]['capacity']
            T_j0 = stations[j]['T_j0']

            p_j_cus = p_j + tax_list[j-1]
            
            # Simplified travel time model based on the provided equation
            # travel_time_j = T_j0 + alpha * max(0, flows[j] - c_j)/c_j
            
            # generalized_costs[j] = p_j + tau * travel_time_j

            # Smooth travel time: softplus instead of max
            congestion = math.log(1 + math.exp(flows[j] - c_j))
            travel_time = T_j0 + alpha * congestion / c_j

            generalized_costs[j] = p_j_cus + tau * travel_time

        # Calculate auxiliary flows using the MNL model
        total_exp_cost = sum(math.exp(-theta * generalized_costs[k]) for k in station_ids)
        auxiliary_flows = {j: demand * math.exp(-theta * generalized_costs[j]) / total_exp_cost for j in station_ids}

        # Update flows with MSA (adaptive step size)
        step_size = min(1.0 / s, step_cap)
        new_flows = {j: flows[j] + step_size * (auxiliary_flows[j] - flows[j])
                     for j in station_ids}

        #Check for convergence
        flow_diff = sum(abs(new_flows[j] - flows[j]) for j in station_ids)
        if flow_diff < convergence_tolerance:
            # print(f"MSA converged in {s} iterations.")
            return new_flows

        flows = new_flows
        
    print("MSA did not converge.", flow_diff)
    return flows