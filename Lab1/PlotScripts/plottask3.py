import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

data = pd.read_csv("outputData.txt", names=[
                   "instance", "algorithm", "random seed", "epsilon", "scale", "threshold", "horizon", "REG", "HIGHS"])
data["algorithm"] = data["algorithm"].str.strip()

algoname = "alg-t3"
horizons = [100, 400, 1600, 6400, 25600, 102400]
instances = ["../instances/instances-task3/i-1.txt", "../instances/instances-task3/i-2.txt"]

averagescores = defaultdict(dict)

for inst in instances:
    for hori in horizons:
        subdata = data.loc[((data["instance"]==inst) & (data["horizon"]==hori))]
        averagescores[inst][hori] = subdata["REG"].mean()

# def pretty(d, indent=0):
#    for key, value in d.items():
#       print('\t' * indent + str(key))
#       if isinstance(value, dict):
#          pretty(value, indent+1)
#       else:
#          print('\t' * (indent+1) + str(value))
# pretty(averagescores)

for ind, inst in enumerate(instances):
    plt.figure(figsize=[8, 6])
    instscores = []
    for h in horizons:
        instscores += [averagescores[inst][h]]
    plt.plot(horizons, instscores, color='b', marker='o', linestyle=':', linewidth=2, alpha=0.8)
    plt.xscale('log')
    plt.title(r'\textbf{alg-t3} --'+'{}'.format(inst), fontsize=18)
    plt.xlabel(r'\textbf{Horizons (log-scale)}', fontsize=16)
    plt.ylabel(r'\textbf{REG (Average Regret over 50 Runs)}', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(f"plot_{ind}.svg")
    plt.show()
