# upper_level_milp.py
import gurobipy as gp
from gurobipy import GRB
from lower_level_sue import solve_sue_msa
from scipy.optimize import minimize


def best_response_station_joint(
    j,
    stations,
    demand,
    theta,
    tau,
    alpha,
    fixed_cost_rate=15,
    operating_cost_rate=15,
    price_bounds=(1, 30),
    capacity_bounds=(1, 30),
):
    """
    Computes the joint best-response (price and capacity) for station j.
    """
    # Initial guess
    x0 = [stations[j]["price"], stations[j]["capacity"]]

    def neg_profit(x):
        p_j, c_j = x
        stations[j]["price"] = p_j
        stations[j]["capacity"] = c_j
        # Solve approximate SUE
        flows = solve_sue_msa(demand, stations, theta, tau, alpha)
        q_j = flows[j]
        wait_time = max(0, q_j - c_j) / c_j
        wait_time_penalty = tau * wait_time
        fixed_cost = fixed_cost_rate * c_j
        operating_cost = operating_cost_rate * q_j
        profit = p_j * q_j - fixed_cost - operating_cost - wait_time_penalty * q_j
        return -profit  # negative for minimization

    bounds = [price_bounds, capacity_bounds]
    res = minimize(neg_profit, x0=x0, bounds=bounds, method='L-BFGS-B') # "SLSQP"

    if res.success:
        p_opt, c_opt = res.x
        return p_opt, c_opt, -res.fun
    else:
        print(f"Best response for station {j} failed: {res.message}")
        return x0[0], x0[1], -neg_profit(x0)


def nash_equilibrium(stations, demand, theta, tau, alpha, fixed_cost_rate, operating_cost_rate, tol=1e-4, max_iter=200):
    for it in range(max_iter):
        max_change = 0

        for j in stations.keys():
            print(f"Iteration {it+1}, Station {j}:")
            print(f"  Current price: {stations[j]['price']:.4f}")
            print(f"  Current capacity: {stations[j]['capacity']}")
            p_old, c_old = stations[j]["price"], stations[j]["capacity"]
            # Joint best response
            p_new, c_new, _ = best_response_station_joint(
                j, stations, demand, theta, tau, alpha, fixed_cost_rate, operating_cost_rate
            )

            """
            Round c_new to nearest integer and compute max change accordingly
            this round may cause oscillations/suboptimality            
            """
            if abs(c_new - c_old) <= 0.5:
                max_capacity_change = 0
            else:
                max_capacity_change = 1

            # max_capacity_change = abs(c_new - c_old)

            max_change = max(max_change, abs(p_new - p_old), max_capacity_change)

            stations[j]["price"] = p_new
            stations[j]["capacity"] = round(c_new)

            print(f"Station {j} updated price to {p_new:.4f}")
            print(f"Station {j} updated capacity to {c_new}")

            print(
                f"  Price change: {abs(p_new - p_old):.6f}, Capacity change: {abs(c_new - c_old), max_capacity_change}",
                max_change,
            )

        if max_change < tol:
            print(f"Nash equilibrium converged in {it+1} iterations. Max change: {max_change:.6f}")
            break

    flows = solve_sue_msa(demand, stations, theta, tau, alpha)
    return stations, flows
