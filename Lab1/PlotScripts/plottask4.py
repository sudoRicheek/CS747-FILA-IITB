import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.rcParams['text.usetex'] = True

data = pd.read_csv("outputData.txt", names=[
                   "instance", "algorithm", "random seed", "epsilon", "scale", "threshold", "horizon", "REG", "HIGHS"])
data["algorithm"] = data["algorithm"].str.strip()

algoname = "alg-t4"
thresholds = [0.2, 0.6]
horizons = [100, 400, 1600, 6400, 25600, 102400]
instances = ["../instances/instances-task4/i-1.txt", "../instances/instances-task4/i-2.txt"]
maxhigh_prob = {
    "../instances/instances-task4/i-1.txt": {0.2: 1.0, 0.6: 0.4},
    "../instances/instances-task4/i-2.txt": {0.2: 0.9, 0.6: 0.53}
}

averagescores = defaultdict(lambda: defaultdict(dict))

for inst in instances:
    for th in thresholds:
        for hori in horizons:
            subdata = data.loc[((data["instance"]==inst) & (data["threshold"]==th) & (data["horizon"]==hori))]
            averagescores[inst][th][hori] = maxhigh_prob[inst][th]*hori - subdata["HIGHS"].mean() 

# def pretty(d, indent=0):
#    for key, value in d.items():
#       print('\t' * indent + str(key))
#       if isinstance(value, dict):
#          pretty(value, indent+1)
#       else:
#          print('\t' * (indent+1) + str(value))
# pretty(averagescores)

# colorm = {"epsilon-greedy-t1": 'r', "ucb-t1": 'b', "kl-ucb-t1": 'g', "thompson-sampling-t1": 'darkorchid'}
for ind, inst in enumerate(instances):
    for th in thresholds:
        plt.figure(figsize=[8, 6])
        algoscores = []
        for h in horizons:
            algoscores += [averagescores[inst][th][h]]
        plt.plot(horizons, algoscores, color='b', marker='o', linestyle=':', linewidth=2, alpha=0.8)
        plt.xscale('log')
        plt.title(r'\textbf{alg-t4} -- '+'{} -- Threshold={}'.format(inst, th), fontsize=18)
        plt.xlabel(r'\textbf{Horizons (log-scale)}', fontsize=16)
        plt.ylabel(r'\textbf{HIGHS-REGRET (Averaged over 50 runs)}', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig(f"plot_{ind}_{th}.svg")
        plt.show()
