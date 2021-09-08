import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

data = pd.read_csv("outputData.txt", names=[
                   "instance", "algorithm", "random seed", "epsilon", "scale", "threshold", "horizon", "REG", "HIGHS"])
data["algorithm"] = data["algorithm"].str.strip()

algonames = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1"]
horizons = [100, 400, 1600, 6400, 25600, 102400]
instances = ["../instances/instances-task1/i-1.txt", "../instances/instances-task1/i-2.txt", "../instances/instances-task1/i-3.txt"]

averagescores = defaultdict(lambda : defaultdict(dict))

for inst in instances:
    for algo in algonames:
        for hori in horizons:
            subdata = data.loc[((data["instance"]==inst) & (data["algorithm"]==algo) & (data["horizon"]==hori))]
            averagescores[inst][algo][hori] = subdata["REG"].mean()

# def pretty(d, indent=0):
#    for key, value in d.items():
#       print('\t' * indent + str(key))
#       if isinstance(value, dict):
#          pretty(value, indent+1)
#       else:
#          print('\t' * (indent+1) + str(value))
# pretty(averagescores)

colorm = {"epsilon-greedy-t1": 'r', "ucb-t1": 'b', "kl-ucb-t1": 'g', "thompson-sampling-t1": 'darkorchid'}
for ind, inst in enumerate(instances):
    plt.figure(figsize=[8, 6])
    for algo in algonames:
        algoscores = []
        for h in horizons:
            algoscores += [averagescores[inst][algo][h]]
        plt.plot(horizons, algoscores, color=colorm[algo], marker='o', linestyle=':', label=algo, linewidth=2, alpha=0.8)
    plt.xscale('log')
    plt.title(r'{}'.format(inst), fontsize=18)
    plt.xlabel(r'\textbf{Horizons (log-scale)}', fontsize=16)
    plt.ylabel(r'\textbf{REG (Average Regret over 50 Runs)}', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend()
    plt.savefig(f"plot_{ind}.svg")
    plt.show()



