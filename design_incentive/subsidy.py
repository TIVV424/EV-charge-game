#%%
#!/usr/bin/env python3
"""
Created on Sept 2025

Author: Ruiting Wang
"""
# add parent dir to path
import sys
mother_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if mother_dir not in sys.path:
    sys.path.insert(0, mother_dir)


import numpy as np
from find_nash.upper_level_milp import nash_equilibrium
from helper_fn.save_results_nash import save_results_full
import os
import copy


def iterative_capacity_subsidy_loop(
    stations,
    demand,
    tau,
    alpha,
    lam=0.6,
    max_capacity=100,
    tol=0.5,
    max_iter=50,
    step_size=1.0,  # step size for fixed cost adjustment
):
    """
    Iteratively adjusts per-unit capacity subsidy (negative fixed cost) to
    equalize travel times across stations.
    """
    M = len(stations)
    # Initialize fixed cost per unit capacity (can be positive or negative)
    fixed_cost_rates = np.array([15.0]*M)
    stations_current = copy.deepcopy(stations)

    for iteration in range(max_iter):
        # Solve Nash equilibrium with current fixed cost rates
        final_stations, flows = nash_equilibrium(
            stations_current,
            demand,
            lam,
            tau,
            alpha,
            fixed_cost_rate=fixed_cost_rates,
            operating_cost_rate=5,
            tax_list=[0]*M,
            tol=1e-3,
            max_iter=500,
            max_capacity=max_capacity,
        )

        # Compute travel times
        T_j = {}
        for j in final_stations:
            q = flows[j]
            c = final_stations[j]["capacity"]
            T0 = final_stations[j]["T_j0"]
            T_j[j] = T0 + alpha * max(0, q - c) / c

        # Target travel time: mean
        T_target = np.mean(list(T_j.values()))

        # Check convergence
        max_dev = max(abs(T_j[j] - T_target) for j in T_j)
        print(f"Iteration {iteration+1}, max deviation = {max_dev:.3f} min")
        if max_dev < tol:
            print("Travel times nearly equalized. Converged.")
            break

        # Update fixed cost rates (subsidy)
        for j in range(M):
            delta = T_j[j+1] - T_target
            # Reduce fixed cost (increase subsidy) if travel time is high
            fixed_cost_rates[j] += -step_size * delta

        print("Updated FIXED_COST_RATES:", fixed_cost_rates)

        # Prepare stations for next iteration
        stations_current = copy.deepcopy(final_stations)

    return final_stations, flows, fixed_cost_rates, T_j

# ---------------- Case Input ----------------
stations = {
    1: {"price": 30, "capacity": 14, "T_j0": 10},
    2: {"price": 30, "capacity": 17, "T_j0": 7},
    3: {"price": 30, "capacity": 19, "T_j0": 5},
    4: {"price": 30, "capacity": 21, "T_j0": 3},
}

demand = 100
tau = 0.5
alpha = 20
lam = 0.6
MAX_CAPACITY = 84

final_stations, final_flows, final_fixed_costs, final_T = iterative_capacity_subsidy_loop(
    stations, demand, tau, alpha, lam, max_capacity=MAX_CAPACITY
)

print("\nFinal Results:")
for j in final_stations:
    print(
        f"Station {j}: Price={final_stations[j]['price']:.2f}, "
        f"Flow={final_flows[j]:.2f}, "
        f"Capacity={final_stations[j]['capacity']}, "
        f"FixedCostRate={final_fixed_costs[j-1]:.2f}, "
        f"TravelTime={final_T[j]:.2f}"
    )


# save
folder = "results_final/subsidy/"
if not os.path.exists(folder):
    os.makedirs(folder)
filename_prefix = folder + f"SUBSIDY_tau{tau}_alpha{alpha}_lam{lam}_demand{demand}"
params = {
    "TOTAL_DEMAND": demand,
    "T_j0": {j: props["T_j0"] for j, props in stations.items()},
    "LAMBDA": lam,
    "TAU": tau,
    "ALPHA": alpha,
    "FINAL_FIXED_COST_RATE": final_fixed_costs.tolist(),
    "MAX_CAPACITY": 84,
}

save_results_full(final_stations, final_flows, params, filename_prefix)
print(f"Results saved to {filename_prefix}_results.json")


# %%
