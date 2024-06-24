import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datasets = ["chameleon", "squirrel", "crocodile"]
data = []
for dataset_name in datasets:
    data.append(pd.read_csv(f"./results_3/{dataset_name}.csv"))

data.append(pd.read_csv(f"./results_3/baselines.csv"))

for i, dataset_name in enumerate(datasets):
    losses = [float(entry.split(" ")[0]) for entry in data[i]["top_10_loss"]]
    errors = [float(entry.split(" ")[2]) for entry in data[i]["top_10_loss"]]
    baseline_loss = float((data[3][(data[3]["dataset"] == dataset_name) & (data[3]["avg_time"] != 0)]["top_10_loss"]).values[0].split(" ")[0])
    baseline_error = float((data[3][(data[3]["dataset"] == dataset_name) & (data[3]["avg_time"] != 0)]["top_10_loss"]).values[0].split(" ")[2])
    xticks = [f"{data[i]['coarsening_ratio'][j]}_{data[i]['extra_nodes'][j]}" for j in range(len(data[i]["coarsening_ratio"]))]

    plt.figure(figsize=(8,8))
    plt.errorbar(np.arange(len(data[i]["coarsening_ratio"])), losses, yerr=errors)
    plt.plot(np.arange(len(data[i]["coarsening_ratio"])), baseline_loss * np.ones(len(data[i]["coarsening_ratio"])))
    plt.title(f"{dataset_name} Top 10 Loss")
    plt.xticks(np.arange(len(data[i]["coarsening_ratio"])), xticks, rotation = 45)
    plt.xlabel("Coarsening Ratio_Extra Nodes")
    plt.ylabel("Top 10 Loss")
    plt.grid(alpha = 0.5)
    plt.legend(["Coarsening", "Baseline"])
    plt.savefig(f"./plots/{dataset_name}_top_10_loss.png")
    plt.close()

    times = [float(entry) for entry in data[i]["ave_time"]]
    baseline_time = float((data[3][(data[3]["dataset"] == dataset_name) & (data[3]["avg_time"] != 0)]["avg_time"]).values[0])

    plt.figure(figsize=(8,8))
    plt.plot(np.arange(len(data[i]["coarsening_ratio"])), times)
    plt.plot(np.arange(len(data[i]["coarsening_ratio"])), baseline_time * np.ones(len(data[i]["coarsening_ratio"])))
    plt.title(f"{dataset_name} Average Time")
    plt.xticks(np.arange(len(data[i]["coarsening_ratio"])), xticks, rotation = 45)
    plt.xlabel("Coarsening Ratio_Extra Nodes")
    plt.ylabel("Average Time")
    plt.grid(alpha = 0.5)
    plt.legend(["Coarsening", "Baseline"])
    plt.savefig(f"./plots/{dataset_name}_average_time.png")
    plt.close()
