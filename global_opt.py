# global_optimization_cooperative_model.py
import gurobipy as gp
from gurobipy import GRB
import json
import pandas as pd
import matplotlib.pyplot as plt


def solve_cooperative_model(demand, stations, fixed_cost_rate, tau, alpha, theta):
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

    prices = model.addVars(station_ids, vtype=GRB.CONTINUOUS, name="price")
    capacities = model.addVars(station_ids, lb = 1, ub = demand, vtype=GRB.INTEGER, name="capacity")
    flows = model.addVars(station_ids, ub = demand, vtype=GRB.CONTINUOUS, name="flow")
    travel_time = model.addVars(station_ids, vtype=GRB.CONTINUOUS, name="travel_time")
    traveler_cost = model.addVars(station_ids,vtype=GRB.CONTINUOUS, name="traveler_cost")
    log_aux = model.addVars(station_ids, lb = 1e-6, vtype=GRB.CONTINUOUS, name="logNBS")
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
        # add some boudnary constraints to avoid numerical issues
        model.addConstr(flows[j] <= 5 + capacities[j], name=f"Capacity_Overflow_Constraint1_{j}")

        # eq (2) travel time function
        model.addConstr(aux_03[j] * capacities[j] == flows[j], name=f"Aux_03_Definition_Constraint_{j}")
        model.addConstr(
            travel_time[j] >= stations[j]["T_j0"] + alpha * (aux_03[j] - 1),
            name=f"Travel_Time_Constraint_{j}",
        )
        model.addConstr(travel_time[j] >= stations[j]["T_j0"], name=f"Min_Travel_Time_Constraint_{j}")
        
        # eq (1) traveler cost function
        model.addConstr( #
            traveler_cost[j] >= prices[j] + tau * travel_time[j], name=f"Traveler_Cost_Definition_Constraint_{j}"
        )

        # NBS component
        model.addConstr(aux_01[j] == O0[j] - traveler_cost[j], name=f"Aux_01_Definition_Constraint_{j}")
        model.addGenConstrLog(aux_01[j], log_aux[j], name="Log_Constraint")


        model.addConstr(aux_02[j] == -1 * THETA * traveler_cost[j],name=f"Aux_02_Definition_Constraint_{j}")
        model.addGenConstrExp(aux_02[j], exp_aux[j], name="Exp_Constraint")

    infra_cost = gp.quicksum(fixed_cost_rate * capacities[j] for j in station_ids)
    NBS_cost = W * gp.quicksum(log_aux[j] for j in station_ids)

    # Constraints
    # q_j = N \cdot \frac{\exp(-\lambda \cdot G_j)}{\sum_{k=1}^{M} \exp(-\lambda \cdot G_k)} \label{eq:demand_allocation}

    denom = model.addVar(vtype=GRB.CONTINUOUS, name="denom")
    model.addConstr(denom == gp.quicksum(exp_aux[j] for j in station_ids))
    model.addConstrs((flows[j] * denom == demand * exp_aux[j] for j in station_ids), name="MNL_reform")
    model.addConstr(gp.quicksum(flows[j] for j in station_ids) == demand, name="Demand_Constraint")

    # Optimize the model
    model.setObjective(infra_cost - NBS_cost, GRB.MINIMIZE)

    model.Params.NonConvex = 2    # required for bilinear/log terms
    model.Params.NumericFocus = 3
    model.Params.BarConvTol = 1e-8
    model.optimize()

    if model.status == GRB.OPTIMAL:
        optimal_solution = {
            "prices": {j: prices[j].X for j in station_ids},
            "capacities": {j: capacities[j].X for j in station_ids},
            "flows": {j: flows[j].X for j in station_ids},
            "travel_time": {j: travel_time[j].X for j in station_ids},
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
    df = pd.DataFrame({
        "Station": station_ids,
        # "Price": [results["prices"][j] for j in station_ids],
        "Capacity": [results["capacities"][j] for j in station_ids],
        "Flow": [results["flows"][j] for j in station_ids],
        "TravelTime": [results["travel_time"][j] for j in station_ids],
        "TravelerCost": [results["traveler_cost"][j] for j in station_ids],
    })
    df.to_csv(f"{filename_prefix}.csv", index=False)
    print(f"✅ Saved solution as {filename_prefix}.json and {filename_prefix}.csv")

    return results, df


def visualize_solution_full(results, title="Globally Optimal Cooperative Solution"):
    stations = list(results["capacities"].keys())


    # Core decision vars
    prices = list(results["prices"].values())
    capacities = list(results["capacities"].values())
    flows = list(results["flows"].values())

    # Time/cost vars
    travel_time = list(results["travel_time"].values())
    traveler_cost = list(results["traveler_cost"].values())

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Prices
    axs[0, 0].bar(stations, prices, color="skyblue")
    axs[0, 0].set_title("Prices")
    axs[0, 0].set_xlabel("Station")
    axs[0, 0].set_ylabel("Price")

    # Capacities
    axs[0, 1].bar(stations, capacities, color="lightgreen")
    axs[0, 1].set_title("Capacities")
    axs[0, 1].set_xlabel("Station")
    axs[0, 1].set_ylabel("Capacity")

    # Flows
    axs[1, 0].bar(stations, flows, color="salmon")
    axs[1, 0].set_title("Flows")
    axs[1, 0].set_xlabel("Station")
    axs[1, 0].set_ylabel("Flow")

    # Travel Time & Traveler Cost (grouped)
    width = 0.35
    x = range(len(stations))
    axs[1, 1].bar([i - width/2 for i in x], travel_time, width, label="Travel Time", color="orange")
    axs[1, 1].bar([i + width/2 for i in x], traveler_cost, width, label="Traveler Cost", color="purple")
    axs[1, 1].set_title("Time & Cost per Station")
    axs[1, 1].set_xlabel("Station")
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Example usage
if __name__ == "__main__":
    TOTAL_DEMAND = 100
    STATIONS = {1: {"T_j0": 5}, 2: {"T_j0": 7}, 3: {"T_j0": 4}, 4: {"T_j0": 3}}
    FIXED_COST_RATE = 1.5
    TAU = 0.5
    ALPHA = 0.5
    THETA = 0.5
    W = 1
    O0 = {j: 100 for j in STATIONS.keys()}  # Example initial utility values

    optimal_solution, model = solve_cooperative_model(TOTAL_DEMAND, STATIONS, FIXED_COST_RATE, TAU, ALPHA, THETA)

    if optimal_solution:
        print("\nGlobally Optimal Cooperative Solution:")
        # print("Optimal Prices:", {j: f"{p:.2f}" for j, p in optimal_solution["prices"].items()})
        print("Optimal Capacities:", {j: f"{c:.2f}" for j, c in optimal_solution["capacities"].items()})
        print("Optimal Flows:", {j: f"{q:.2f}" for j, q in optimal_solution["flows"].items()})

        station_ids = list(STATIONS.keys())
        results, df = save_solution_full(model, station_ids, filename_prefix="demo_case_full_solution")

        # Visualize
        visualize_solution_full(results)

# %%
results, df = save_solution_full(model, station_ids, filename_prefix="demo_case_full_solution")



# %%
