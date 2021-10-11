import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

data = pd.read_csv("outputData.txt", names=[
                   "instance", "algorithm", "random seed", "epsilon", "scale", "threshold", "horizon", "REG", "HIGHS"])
data["algorithm"] = data["algorithm"].str.strip()

algoname = "ucb-t2"
horizon = 10000
scales = list(np.arange(0.02, 0.32, 0.02).round(2))
instances = ["../instances/instances-task2/i-{}.txt".format(i) for i in range(1, 6)]

averagescores = defaultdict(dict)

for inst in instances:
    for c in scales:
        subdata = data.loc[((data["instance"]==inst) & (data["scale"]==c))]
        averagescores[inst][c] = subdata["REG"].mean()

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))
pretty(averagescores)

# plt.figure(figsize=[8, 6])
# for inst in instances:
#     instscores = []
#     for c in scales:
#         instscores += [averagescores[inst][c]]
#     plt.plot(scales, instscores, marker='o', linestyle=':', label=inst, linewidth=2, alpha=0.8)
# plt.title(r'Task2: Regrets vs Scales in different instances', fontsize=18)
# plt.xlabel(r'\textbf{Scale (c)}', fontsize=16)
# plt.ylabel(r'\textbf{REG (Average Regret over 50 Runs)}', fontsize=16)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.legend()
# plt.savefig(f"plot_t2.svg")
# plt.show()



