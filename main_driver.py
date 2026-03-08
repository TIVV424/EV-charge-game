# %%
# main_driver.py
import copy
from lower_level_sue import solve_sue_msa
from upper_level_milp import nash_equilibrium
from save_results_nash import save_results_full
import numpy as np
import pandas as pd
import json
import os

folder = "results_final/nash/"
if not os.path.exists(folder):
    os.makedirs(folder)

# Problem parameters and initial state
NUM_COMPANIES = 4
TOTAL_DEMAND = 100
LAMBDA = 0.6  # scale parameter for logit model
TAU = 0.5  # cost per unit time (min)
ALPHA = 20  # congestion factor - converting to minutes
FIXED_COST_RATE = [10.51, 16.07, 15.48, 17.94]  # fixed cost per unit capacity
OPERATING_COST_RATE = 5  # operating cost per unit flow
TAX_LIST = [0,0,0,0]  # tax/subsidy for each company


# TAX_LIST = [0, 0, 0, 0]  # tax/subsidy for each company
# T_j0: free-flow travel time for each station, unit value is minutes
MAX_CAPACITY = 84
initial_stations = {
    1: {"price": 30, "capacity": 14, "T_j0": 10, "company": 1},
    2: {"price": 30, "capacity": 17, "T_j0": 7, "company": 2},
    3: {"price": 30, "capacity": 19, "T_j0": 5, "company": 3},
    4: {"price": 30, "capacity": 21, "T_j0": 3, "company": 4},
}

filename_prefix = (
    folder
    + f"NASH_lam{LAMBDA}_tau{TAU}_alpha{ALPHA}_oper{OPERATING_COST_RATE}_fix{FIXED_COST_RATE}_maxcap{MAX_CAPACITY}_tax{'_'.join([str(t) for t in TAX_LIST])}"
)

params = {
    "TOTAL_DEMAND": TOTAL_DEMAND,
    "T_j0": {j: props["T_j0"] for j, props in initial_stations.items()},
    "LAMBDA": LAMBDA,
    "TAU": TAU,
    "ALPHA": ALPHA,
    "FIXED_COST_RATE": FIXED_COST_RATE,
    "OPERATING_COST_RATE": OPERATING_COST_RATE,
    "TAX_LIST": TAX_LIST,
    "MAX_CAPACITY": MAX_CAPACITY,
}


final_stations, final_flows = nash_equilibrium(
    initial_stations,
    TOTAL_DEMAND,
    LAMBDA,
    TAU,
    ALPHA,
    FIXED_COST_RATE,
    OPERATING_COST_RATE,
    TAX_LIST,
    tol=1e-3,
    max_iter=5000,
    max_capacity=MAX_CAPACITY,
)

# round capacity
for j in final_stations:
    final_stations[j]["capacity"] = int(round(final_stations[j]["capacity"]))

save_results_full(final_stations, final_flows, params, filename_prefix)

print("Final Prices & Flows:")
for j in final_stations:
    print(
        f"Station {j}: Price = {final_stations[j]['price']:.2f}, \
        Flow = {final_flows[j]:.2f}, \
        Capacity = {final_stations[j]['capacity']}"
    )

# %%
