# %%
# main_driver.py
import copy
from lower_level_sue import solve_sue_msa
from upper_level_milp import nash_equilibrium
import numpy as np
import pandas as pd

# Problem parameters and initial state
NUM_COMPANIES = 4
TOTAL_DEMAND = 100
THETA = 1 # scale parameter for logit model
TAU = 0.5 # cost per unit time (min)
ALPHA = 20 # congestion factor - converting to minutes
FIXED_COST_RATE = 15 # fixed cost per unit capacity
OPERATING_COST_RATE = 10 # operating cost per unit flow

# T_j0: free-flow travel time for each station, unit value is minutes
initial_stations = {
    1: {"price": 2, "capacity": 10, "T_j0": 7, "company": 1},
    2: {"price": 2, "capacity": 40, "T_j0": 10, "company": 2},
    3: {"price": 2, "capacity": 10, "T_j0": 5, "company": 3},
    4: {"price": 2, "capacity": 10, "T_j0": 3, "company": 4},
}


def run_diagonalization_algorithm(convergence_tolerance=1e-3, max_iter=20):
    """
    Runs the diagonalization algorithm to find the Nash equilibrium.
    """
    current_stations = copy.deepcopy(initial_stations)

    for g in range(max_iter):
        print(f"\n--- Diagonalization Iteration {g+1} ---")
        prev_prices = {j: data["price"] for j, data in current_stations.items()}

        # Iterate through each company to find its best response
        for n in range(1, NUM_COMPANIES + 1):
            print(f"  Solving for Company {n}...")

            # Step 1: Solve the lower-level SUE problem
            equilibrium_flows = solve_sue_msa(
                demand=TOTAL_DEMAND, stations=current_stations, theta=THETA, tau=TAU, alpha=ALPHA
            )

            # Step 2: Solve the upper-level MILP for company n
            optimized_price, optimized_capacity = solve_company_milp(
                company_id=n,
                stations=current_stations,
                satisfied_demands=equilibrium_flows,
                fixed_cost_rate=FIXED_COST_RATE,
                operating_cost_rate=OPERATING_COST_RATE,
            )

            # Handle cases where Gurobi might not find a solution
            if optimized_price is None:
                print(f"  Company {n} could not find an optimal solution. Stopping.")
                return current_stations

            # Step 3: Update company n's strategy
            for j, data in current_stations.items():
                if data["company"] == n:
                    data["price"] = optimized_price
                    data["capacity"] = optimized_capacity

            print(f" Company {n} updated its price to {optimized_price:.2f} and capacity to {optimized_capacity}.")

        # Step 4: Check for convergence
        current_prices = {j: data["price"] for j, data in current_stations.items()}
        price_diff = sum(abs(current_prices[j] - prev_prices[j]) for j in current_stations)

        if price_diff < convergence_tolerance:
            print("\nDiagonalization algorithm converged!")
            return current_stations

    print("\nDiagonalization algorithm did not converge within the maximum iterations.")
    return current_stations


# save results
def save_results(stations, flows, filename="nash_equilibrium_results.csv"):
    # flow, price, capacity, wait time, travel time, cost
    data = []
    for j, props in stations.items():
        p_j = props["price"]
        c_j = props["capacity"]
        T_j0 = props["T_j0"]
        q_j = flows[j]

        # travel time
        congestion = (q_j - c_j) if q_j > c_j else 0
        travel_time = T_j0 + ALPHA * congestion / c_j

        # cost
        cost = p_j + TAU * travel_time

        data.append(
            {"Station": j, 
             "Price": p_j, 
             "Capacity": c_j, 
             "Flow": q_j, 
             "TravelTime": travel_time, 
             "TravelerCost": cost}
        )

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


final_stations, final_flows = nash_equilibrium(initial_stations, TOTAL_DEMAND, THETA, TAU, ALPHA)

save_results(final_stations, final_flows)

print("Final Prices & Flows:")
for j in final_stations:
    print(
        f"Station {j}: Price = {final_stations[j]['price']:.2f}, \
        Flow = {final_flows[j]:.2f}, \
        Capacity = {final_stations[j]['capacity']}"
    )


# %%
