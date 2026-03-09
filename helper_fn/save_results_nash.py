import pandas as pd
import json

def save_results_full(stations, flows, params, filename_prefix="nash_equilibrium_results"):
    """
    Save Nash equilibrium results in JSON + CSV format.
    """
    results = {
        "prices": {},
        "capacities": {},
        "flows": {},
        "travel_time": {},
        "traveler_cost": {},
    }

    data = []
    for j, props in stations.items():
        p_j = props["price"]
        c_j = props["capacity"]
        T_j0 = props["T_j0"]
        q_j = flows[j]

        # Travel time (with congestion penalty)
        congestion = (q_j - c_j) if q_j > c_j else 0
        congestion_time = params["ALPHA"] * congestion / c_j
        travel_time = T_j0 + congestion_time

        # Traveler cost
        cost = p_j + params["TAU"] * travel_time

        # Store in results dict
        results["prices"][j] = p_j
        results["capacities"][j] = c_j
        results["flows"][j] = q_j
        results["travel_time"][j] = travel_time
        results["traveler_cost"][j] = cost

        # Store row for CSV
        data.append({
            "Station": j,
            "Price": p_j,
            "Capacity": c_j,
            "Flow": q_j,
            "TravelTime": travel_time,
            "TravelerCost": cost,
            "TotalDemand": params["TOTAL_DEMAND"],
        })

    # Merge with params for JSON output
    results.update(params)

    # Save JSON
    with open(f"{filename_prefix}.json", "w") as f:
        json.dump(results, f, indent=4)

    # Save CSV
    df = pd.DataFrame(data)
    df.to_csv(f"{filename_prefix}.csv", index=False)

    print(f"Saved results as {filename_prefix}.json and {filename_prefix}.csv")

    return results, df

