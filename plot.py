import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv")
for dataset in df["dataset"].unique():
    plt.figure()
    plt.plot(df[df["dataset"] == dataset]["coarsening_ratio"], np.log10(df[df["dataset"] == dataset]["sum_squares_subgraph_nodes"]), label = 'Ours', marker = "o", color = 'b')
    plt.plot(df[df["dataset"] == dataset]["coarsening_ratio"], np.log10(df[df["dataset"] == dataset]["tot_num_orig_nodes_squared"]), label = 'Original', marker = "^", color = 'g' )
    plt.title(f"{dataset}\nSum of squares of subgraph nodes vs Coarsening Ratio" )
    plt.xlabel("Coarsening Ratio")
    plt.ylabel("Sum of squares of subgraph nodes (log scale)")
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.savefig(f"./plots/{dataset}_sum_squares_subgraph_nodes.png")

for dataset in df["dataset"].unique():
    plt.figure()
    plt.plot(df[df["dataset"] == dataset]["coarsening_ratio"], np.log10(df[df["dataset"] == dataset]["total"]), marker = "o", color = 'b')
    if dataset in ["dblp", "Physics"]:
        plt.title(f"{dataset} - Total space occupied* vs Coarsening Ratio\n*Not including sizes of train, val and test masks" )
    else:
        plt.title(f"{dataset}\nTotal space occupied vs Coarsening Ratio" )
    plt.xlabel("Coarsening Ratio")
    plt.ylabel("Total space occupied (in log10(byes))")
    plt.grid(alpha = 0.5)
    plt.savefig(f"./plots/{dataset}_total_space.png")
