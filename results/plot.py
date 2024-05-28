import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cora.csv')
orig = pd.read_csv("results_orig.csv")
orig = orig[orig['dataset'] == "cora"]
orig['top_10_acc'] = orig.iloc[:, 13:23].mean(axis = 1)
orig.reset_index(drop = True, inplace = True)
orig = orig.iloc[[0, 5, 1, 6, 2, 7, 3, 8, 4, 9],:]


for parameter in ["ave_acc", "top_10_acc"]:
    
    y1 = df[df['exp_setup'] == "Gc_train_2_Gs_train"][parameter]
    y1 = [float(e.split(' ')[0]) for e in y1]

    y2 = df[df['exp_setup'] == "Gc_train_2_Gs_infer"][parameter]
    y2 = [float(e.split(' ')[0]) for e in y2]

    y3 = df[df['exp_setup'] == "Gs_train_2_Gs_infer"][parameter]
    y3 = [float(e.split(' ')[0]) for e in y3]

    text = []
    coarsening_ratios = [0.0, 0.0, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7]
    for i in range(len(coarsening_ratios)):
        if i % 2 == 0:
            text.append(str(coarsening_ratios[i])+"-"+" fixed")
        else:
            text.append(str(coarsening_ratios[i])+"-"+" few")

    plt.figure()
    plt.plot(np.arange(len(df[df['exp_setup'] == "Gc_train_2_Gs_train"])), y1, 'r-^', label = "Gc_train_2_Gs_train")
    plt.plot(np.arange(len(df[df['exp_setup'] == "Gc_train_2_Gs_infer"])), y2, 'g-o', label = "Gc_train_2_Gs_infer")
    plt.plot(np.arange(len(df[df['exp_setup'] == "Gs_train_2_Gs_infer"])), y3, 'b-*', label = "Gs_train_2_Gs_infer")
    plt.plot(np.arange(len(orig)), orig[parameter], 'black', linestyle = 'dashed', marker = "X", label = "Original")
    plt.title(f'{parameter}')
    plt.ylabel("Accuracy")
    for i in range(len(coarsening_ratios)):
        plt.text(i, 0.4, text[i], rotation = 90)
    plt.legend()
    plt.grid(alpha = 0.5)
    plt.savefig(f'{parameter}.png')