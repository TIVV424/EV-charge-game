import matplotlib.pyplot as plt
import seaborn as sns
"""
Three subplots are included:
1. (clustered bar) Capacitys, Flows, (lines) Infrastructure Cost, Price
2. individual (stacked bar) ori time cost, congested time cost, price
3. total cost, culmulative (stacked bar) infra cost, ori time cost, congested time cost, price
"""

def visualize_solution(results, title="Globally Optimal Cooperative Solution"):
    stations = list(results["capacities"].keys())
    infra_cost = results["FIXED_COST_RATE"]
    prices = list(results["prices"].values())
    capacities = list(results["capacities"].values())
    flows = list(results["flows"].values())
    travel_time = list(results["travel_time"].values())
    traveler_cost = list(results["traveler_cost"].values())
    T_j0 = list(results["T_j0"].values())

    original_travel_time = [T_j0[i] for i in range(len(stations))]
    congest_travel_time = [travel_time[i] - original_travel_time[i] for i in range(len(stations))]
    

    colormap = sns.color_palette("Set2", 8)
    
    fig, ax = plt.subplots(3, 1, figsize=(3, 6), sharex=True)
    ax = ax.ravel()

    # Subplot 1: Capacities, Flows, Infrastructure Cost, Price
    width = 0.3
    x = range(len(stations))
    ax[0].bar([i - width/2 for i in x], capacities, width, label="Capacity", color = colormap[0])
    ax[0].bar([i + width/2 for i in x], flows, width, label="Flow", color = colormap[1])

    # ax2 = ax[0].twinx()
    # ax2.plot(stations, prices, label="Price", color = colormap[2], marker='o')
    # ax2.set_ylabel("Price")

    ax[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    # Subplot 2: Individual Time Breakdown
    ax[1].bar(stations, original_travel_time, label="Original Travel Time", color = colormap[3], width=0.4)
    ax[1].bar(stations, congest_travel_time, bottom=original_travel_time, label="Congested Travel Time", color = colormap[4], width=0.4)
    ax[1].plot(stations, travel_time, label="Individual Traveler Time", color = colormap[5], marker='o')

    for i in range(len(stations)):
        ax[1].text(stations[i], original_travel_time[i]/2, f"{original_travel_time[i]:.1f}", ha='center', va='center', color='black', fontsize =8)
        ax[1].text(stations[i], original_travel_time[i] + congest_travel_time[i]/2, f"{congest_travel_time[i]:.1f}", ha='center', va='center', color='black', fontsize =8)
        ax[1].text(stations[i], travel_time[i]+1, f"{travel_time[i]:.1f}", ha='center', va='bottom', color='black', fontsize =8)
    
    ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    # Subplot 3: Cumulative Cost Breakdown
    # price, original time cost, congested time cost, infra cost

    ax[2].bar(stations, prices,
                label="Price Paid", color = colormap[2], width=0.4)
    ax[2].bar(stations, [original_travel_time[i] * results["TAU"] for i in range(len(stations))],
                bottom=[prices[i] for i in range(len(stations))], label="Original Time Cost", color = colormap[3], width=0.4)
    bottom_cum = [prices[i] + original_travel_time[i] * results["TAU"] for i in range(len(stations))]
    ax[2].bar(stations, [congest_travel_time[i] * results["TAU"] for i in range(len(stations))],
                bottom=bottom_cum, label="Congested Time Cost", color = colormap[4] ,width=0.4)
    


    # infra_cost_total = [capacities[i] * infra_cost for i in range(len(stations))]
    # # ax[2].bar(stations, infra_cost_total, label="Infrastructure Cost", color = colormap[6], width=0.4)
    # ax[2].bar(stations, [flows[i] * original_travel_time[i] * results["TAU"] for i in range(len(stations))],
    #             label="Original Time Cost", color = colormap[3], width=0.4)tr
    # bottom_cum = [flows[i] * original_travel_time[i] * results["TAU"] for i in range(len(stations))]
    # ax[2].bar(stations, [flows[i] * congest_travel_time[i] * results["TAU"] for i in range(len(stations))],
    #             bottom=bottom_cum, label="Congested Time Cost", color = colormap[4] ,width=0.4)
    # bottom_cum = [bottom_cum[i] + flows[i] * congest_travel_time[i] * results["TAU"] for i in range(len(stations))]
    # ax[2].bar(stations, [flows[i] * prices[i] for i in range(len(stations))],
    #             bottom=bottom_cum, label="Price Paid", color = colormap[2], width=0.4)

    ax[2].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))


    ax[0].set_ylabel("Capacity / Flow")
    ax[1].set_ylabel("Time (min)")
    ax[2].set_xlabel("Station")
    ax[2].set_ylabel("Cost")

    ax[2].set_title("Cumulative Cost Breakdown", x = 0.05, y= 0.8, loc = 'left', fontsize =10)
    # ax[2].set_ylim(0, 800)
    ax[0].set_title("Capacities and Flows",  x = 0.05, y= 0.8, loc = 'left', fontsize =10)
    ax[1].set_title("Traveler Time Breakdown",  x = 0.05, y= 0.8, loc = 'left', fontsize =10)
    ax[0].set_ylim(0, max(flows)*1.5)
    ax[1].set_ylim(0, max(travel_time)*1.5)
    ax[2].set_ylim(0, max([prices[i] + travel_time[i]*results["TAU"] for i in range(len(stations))])*1.5)


    # ax[0].set_xlim(0.5, 4.5) 
    ax[0].set_xticks(['1', '2', '3', '4'])

    plt.savefig(title, bbox_inches='tight', dpi=300)

    return 0




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
    axs[1, 1].bar([i - width / 2 for i in x], travel_time, width, label="Travel Time", color="orange")
    axs[1, 1].bar([i + width / 2 for i in x], traveler_cost, width, label="Traveler Cost", color="purple")
    axs[1, 1].set_title("Time & Cost per Station")
    axs[1, 1].set_xlabel("Station")
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
