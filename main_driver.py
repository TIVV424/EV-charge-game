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
OPERATING_COST_RATE = 15 # operating cost per unit flow

# T_j0: free-flow travel time for each station, unit value is minutes
initial_stations = {
    1: {"price": 2, "capacity": 10, "T_j0": (7+20), "company": 1},
    2: {"price": 2, "capacity": 40, "T_j0": (10+20), "company": 2},
    3: {"price": 2, "capacity": 10, "T_j0": (5+20), "company": 3},
    4: {"price": 2, "capacity": 10, "T_j0": (3+20), "company": 4},
}


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
