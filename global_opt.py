# global_optimization_cooperative_model.py
import gurobipy as gp
from gurobipy import GRB
import json
import pandas as pd
import matplotlib.pyplot as plt
from plot_result import visualize_solution_full

def solve_cooperative_model(demand, stations, fixed_cost_rate, tau, alpha, lam):
    """
    Finds the global optimal solution by minimizing total system cost.

    Args:
        demand (float): Total number of travelers (N).
        stations (dict): A dictionary of stations with their properties (T_j0).
        fixed_cost_rate (float): Fixed cost per unit of capacity.
        operating_cost_rate (float): Operating cost per unit of flow.
        tau (float): Travel time cost coefficient.
        alpha (float): Congestion sensitivity coefficient.

    Returns:
        tuple: A dictionary of optimal prices and capacities for all stations.
    """

    model = gp.Model("Global_Cooperative_Optimization")

    # Decision variables for all stations
    station_ids = list(stations.keys())

    prices = model.addVars(station_ids, lb=0.1, ub=50, vtype=GRB.CONTINUOUS, name="price")
    capacities = model.addVars(station_ids, lb=1, ub=demand / 3, vtype=GRB.INTEGER, name="capacity")
    flows = model.addVars(station_ids, ub=demand, vtype=GRB.CONTINUOUS, n8ame="flow")
    travel_time = model.addVars(station_ids, vtype=GRB.CONTINUOUS, name="travel_time")
    traveler_cost = model.addVars(station_ids, vtype=GRB.CONTINUOUS, name="traveler_cost")
    log_aux = model.addVars(station_ids, lb=1e-6, vtype=GRB.CONTINUOUS, name="logNBS")
    exp_aux = model.addVars(station_ids, vtype=GRB.CONTINUOUS, name="expMNL")
    aux_01 = model.addVars(station_ids, vtype=GRB.CONTINUOUS, name="aux_01")
    aux_02 = model.addVars(station_ids, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="aux_02")
    aux_03 = model.addVars(station_ids, vtype=GRB.CONTINUOUS, name="aux_03")

    # Objective Function
    # The objective is the sum of travelers' costs and operators' costs.
    # Note: Gurobi requires non-linear terms to be expressed carefully. Here we use an approximation
    # or assume a linear relationship for simplicity. A more complex model would use
    # Gurobi's non-linear capabilities or a piecewise linear approximation.

    # Travel time for each station
    for j in station_ids:
        # eq (2) travel time function
        model.addConstr(aux_03[j] * capacities[j] == (flows[j] - capacities[j]), name=f"Aux_03_Definition_Constraint_{j}")
        model.addConstr(
            travel_time[j] >= stations[j]["T_j0"] + alpha * aux_03[j],
            name=f"Travel_Time_Constraint_{j}",
        )
        model.addConstr(travel_time[j] >= stations[j]["T_j0"], name=f"Min_Travel_Time_Constraint_{j}")

        # eq (1) traveler cost function
        model.addConstr(  #
            traveler_cost[j] == prices[j] + tau * travel_time[j], name=f"Traveler_Cost_Definition_Constraint_{j}"
        )

        # NBS component # flows[j]
        model.addConstr(aux_01[j] ==  (O0[j] - traveler_cost[j]), name=f"Aux_01_Definition_Constraint_{j}")
        model.addGenConstrLog(aux_01[j], log_aux[j], name="Log_Constraint")

        # SUE component
        model.addConstr(aux_02[j] == -1 * lam * traveler_cost[j], name=f"Aux_02_Definition_Constraint_{j}")
        model.addGenConstrExp(aux_02[j], exp_aux[j], name="Exp_Constraint")

    infra_cost = gp.quicksum(fixed_cost_rate * capacities[j] for j in station_ids)
    SUM_TIME_cost = tau * gp.quicksum(flows[j] * travel_time[j] for j in station_ids)
    SUM_CUS_UTILITY = gp.quicksum(traveler_cost[j] * flows[j] for j in station_ids)
    PROFIT = gp.quicksum(prices[j] * flows[j] for j in station_ids)


    NBS_cost = W * gp.quicksum(log_aux[j] for j in station_ids)
    NBS_obj = infra_cost - NBS_cost

    Uti_obj_wo_pro = SUM_TIME_cost + infra_cost
    Uti_obj_wi_pro = SUM_TIME_cost + infra_cost - PROFIT
    Customer_obj = SUM_CUS_UTILITY + infra_cost

    # Constraints
    # q_j = N \cdot \frac{\exp(-\lam \cdot G_j)}{\sum_{k=1}^{M} \exp(-\lam \cdot G_k)} \label{eq:demand_allocation}

    denom = model.addVar(vtype=GRB.CONTINUOUS, name="denom")
    model.addConstr(denom == gp.quicksum(exp_aux[j] for j in station_ids))
    model.addConstrs((flows[j] * denom == demand * exp_aux[j] for j in station_ids), name="MNL_reform")
    model.addConstr(gp.quicksum(flows[j] for j in station_ids) == demand, name="Demand_Constraint")

    # Optimize the model
    model.setObjective(NBS_cost, GRB.MINIMIZE)

    model.Params.NonConvex = 2  # required for bilinear/log terms
    model.Params.TimeLimit = 200  # 5 minutes
    model.Params.MIPGap = 0.02
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
        optimal_solution = {
            "prices": {j: prices[j].X for j in station_ids},
            "capacities": {j: capacities[j].X for j in station_ids},
            "flows": {j: flows[j].X for j in station_ids},
            "travel_time": {j: travel_time[j].X for j in station_ids},
            "traveler_cost": {j: traveler_cost[j].X for j in station_ids},
        }
        return optimal_solution, model
    else:
        # print sol status
        print(f"Optimization ended with status {model.status}")
        print("Model did not find an optimal solution.")
        # save which constraints are violated

        model.computeIIS()
        model.write("infeasible_IIS.ilp")

        return None


def save_solution_full(model, station_ids, filename_prefix="solution_full"):
    """
    Extract all decision variables from Gurobi model and save as JSON + CSV.
    """
    # Collect values
    results = {
        "prices": {j: model.getVarByName(f"price[{j}]").X for j in station_ids},
        "capacities": {j: model.getVarByName(f"capacity[{j}]").X for j in station_ids},
        "flows": {j: model.getVarByName(f"flow[{j}]").X for j in station_ids},
        "travel_time": {j: model.getVarByName(f"travel_time[{j}]").X for j in station_ids},
        "traveler_cost": {j: model.getVarByName(f"traveler_cost[{j}]").X for j in station_ids},
    }

    # Save JSON
    with open(f"{filename_prefix}.json", "w") as f:
        json.dump(results, f, indent=4)

    # Save CSV (flatten into dataframe)
    df = pd.DataFrame(
        {
            "Station": station_ids,
            "Price": [results["prices"][j] for j in station_ids],
            "Capacity": [results["capacities"][j] for j in station_ids],
            "Flow": [results["flows"][j] for j in station_ids],
            "TravelTime": [results["travel_time"][j] for j in station_ids],
            "TravelerCost": [results["traveler_cost"][j] for j in station_ids],
        }
    )
    df.to_csv(f"{filename_prefix}.csv", index=False)
    print(f"✅ Saved solution as {filename_prefix}.json and {filename_prefix}.csv")

    return results, df




# Example usage
if __name__ == "__main__":
    TOTAL_DEMAND = 100
    STATIONS = {1: {"T_j0": 6 + 20}, 2: {"T_j0": 5 + 20}, 3: {"T_j0": 4 + 20}, 4: {"T_j0": 3 + 20}}
    LAMBDA = 0.6  # scale parameter for logit model
    TAU = 0.2  # cost per unit time (min)
    ALPHA = 20  # congestion factor - converting to minutes
    FIXED_COST_RATE = 15  # fixed cost per unit capacity
    OPERATING_COST_RATE = 10  # operating cost per unit flow
    W = 100
    O0 = {j: 100 for j in STATIONS.keys()}  # Example initial utility values

    optimal_solution, model = solve_cooperative_model(TOTAL_DEMAND, STATIONS, FIXED_COST_RATE, TAU, ALPHA, LAMBDA)

    if optimal_solution:
        print("\nGlobally Optimal Cooperative Solution:")
        print("Optimal Prices:", {j: f"{p:.2f}" for j, p in optimal_solution["prices"].items()})
        print("Optimal Capacities:", {j: f"{c:.2f}" for j, c in optimal_solution["capacities"].items()})
        print("Optimal Flows:", {j: f"{q:.2f}" for j, q in optimal_solution["flows"].items()})
        print("Optimal Travel Times:", {j: f"{tt:.2f}" for j, tt in optimal_solution["travel_time"].items()})
        print("Optimal Traveler Costs:", {j: f"{tc:.2f}" for j, tc in optimal_solution["traveler_cost"].items()})

        station_ids = list(STATIONS.keys())
        results, df = save_solution_full(model, station_ids, filename_prefix="demo_case_full_solution")

        # Visualize
        visualize_solution_full(results)

# %%
results, df = save_solution_full(model, station_ids, filename_prefix="demo_case_full_solution")
