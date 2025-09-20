# upper_level_milp.py
import gurobipy as gp
from gurobipy import GRB
from lower_level_sue import solve_sue_msa
from scipy.optimize import minimize


def solve_company_milp(company_id, stations, satisfied_demands, fixed_cost_rate, operating_cost_rate):

    """
    Solves the Mixed-Integer Linear Program (MILP) for a single company to maximize profit.
    
    Args:
        company_id (int): ID of the company to optimize.
        stations (dict): A dictionary of all stations.
        satisfied_demands (dict): The satisfied demand for each station from the SUE model.
        fixed_cost_rate (float): Fixed cost per unit of capacity.
        operating_cost_rate (float): Operating cost per unit of flow.
        
    Returns:
        tuple: Optimal price and capacity for the company.
    """
    model = gp.Model(f"Company_{company_id}_MILP")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output

    # Decision variables for the company being optimized
    price = model.addVar(lb=0.01, vtype=GRB.CONTINUOUS, name="price")
    capacity = model.addVar(lb=1, vtype=GRB.INTEGER, name="capacity")
    
    # Get the stations belonging to the company being optimized
    company_stations = {j: data for j, data in stations.items() if data['company'] == company_id}
    station_ids = list(company_stations.keys())
    
    # Define profit components
    total_satisfied_demand = sum(satisfied_demands[j] for j in station_ids)
    
    revenue = price * total_satisfied_demand
    fixed_cost = fixed_cost_rate * capacity
    operating_cost = operating_cost_rate * total_satisfied_demand
    
    # Objective function: Maximize profit
    model.setObjective(revenue - fixed_cost - operating_cost, GRB.MAXIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return price.X, capacity.X
    else:
        print(f"Company {company_id} MILP did not find an optimal solution.")
        print(f" Gurobi status code: {model.status}")
        print(company_id, stations, satisfied_demands, fixed_cost_rate, operating_cost_rate)
        return None, None
    


def best_response_station_joint(j, stations, demand, theta, tau, alpha,
                                price_bounds=(0.01, 10), capacity_bounds=(1, 20),
                                fixed_cost_rate=1.5, operating_cost_rate=0.8):
    """
    Computes the joint best-response (price and capacity) for station j.
    """
    # Initial guess
    x0 = [stations[j]['price'], stations[j]['capacity']]
    
    def neg_profit(x):
        p_j, c_j = x
        stations[j]['price'] = p_j
        stations[j]['capacity'] = c_j
        # Solve approximate SUE
        flows = solve_sue_msa(demand, stations, theta, tau, alpha)
        q_j = flows[j]
        fixed_cost = fixed_cost_rate * c_j
        operating_cost = operating_cost_rate * q_j
        profit = p_j * q_j - fixed_cost - operating_cost
        return -profit  # negative for minimization
    
    bounds = [price_bounds, capacity_bounds]
    res = minimize(neg_profit, x0=x0, bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        p_opt, c_opt = res.x
        return p_opt, c_opt, -res.fun
    else:
        print(f"Best response for station {j} failed: {res.message}")
        return x0[0], x0[1], -neg_profit(x0)

def best_response_station_milp(j, stations, flows, fixed_cost_rate=1.5, operating_cost_rate=0.8):
    """
    Computes best-response (price and integer capacity) for station j using MILP,
    assuming flows from SUE are fixed.
    """
    model = gp.Model(f"Station_{j}_MILP")
    model.Params.OutputFlag = 0  # silent

    # Decision variables
    p = model.addVar(lb=0.01, ub=10, vtype=GRB.CONTINUOUS, name="price")
    c = model.addVar(lb=1, ub=30, vtype=GRB.INTEGER, name="capacity")

    # Fixed flow for this iteration (from last SUE)
    q_j_fixed = flows[j]

    # Linearized profit: revenue - cost
    revenue = p * q_j_fixed
    fixed_cost = fixed_cost_rate * c
    operating_cost = operating_cost_rate * q_j_fixed
    profit = revenue - fixed_cost - operating_cost

    # Objective
    model.setObjective(profit, GRB.MAXIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return p.X, c.X, model.objVal
    else:
        print(f"Station {j} MILP failed: status {model.status}")
        return stations[j]['price'], stations[j]['capacity'], -1
    

def nash_equilibrium(stations, demand, theta, tau, alpha, tol=1e-4, max_iter=200):
    for it in range(max_iter):
        max_change = 0

        for j in stations.keys():
            print(f"Iteration {it+1}, Station {j}:")
            print(f"  Current price: {stations[j]['price']:.4f}")
            print(f"  Current capacity: {stations[j]['capacity']}")
            p_old, c_old = stations[j]['price'], stations[j]['capacity']
            # Joint best response
            p_new, c_new, _ = best_response_station_joint(j, stations, demand, theta, tau, alpha)
            
            if abs(c_new - c_old) <= 0.5:
                max_capacity_change = 0
            else:
                max_capacity_change = 1
            
            max_change = max(max_change, abs(p_new - p_old), max_capacity_change)

            stations[j]['price'] = p_new
            stations[j]['capacity'] = round(c_new)

            print(f"Station {j} updated price to {p_new:.4f}")
            print(f"Station {j} updated capacity to {c_new}")

            print(f"  Price change: {abs(p_new - p_old):.6f}, Capacity change: {abs(c_new - c_old), max_capacity_change}", max_change)

        if max_change < tol:
            print(f"Nash equilibrium converged in {it+1} iterations. Max change: {max_change:.6f}")
            break
    flows = solve_sue_msa(demand, stations, theta, tau, alpha)
    return stations, flows
