import argparse
import numpy as np
from numpy.random import default_rng

parser = argparse.ArgumentParser(description='Bandit Instance')
parser.add_argument('--instance', type=str, help='where in is a path to the instance file')
parser.add_argument('--algorithm', type=str, help='where al is one of epsilon-greedy-t1, ucb-t1, kl-ucb-t1, thompson-sampling-t1, ucb-t2, alg-t3, alg-t4')
parser.add_argument('--randomSeed', type=int, help='where rs is a non-negative integer', default=42)
parser.add_argument('--epsilon', type=float, help='where ep is a number in [0, 1]. For everything except epsilon-greedy, pass 0.02', default=0.02)
parser.add_argument('--scale', type=float, help='where c is a positive real number. The parameter is only relevant for Task 2; for other tasks pass --scale 2', default=2)
parser.add_argument('--threshold', type=float, help='where th is a number in [0, 1]. The parameter is only relevant for Task 4; for other tasks pass --threshold 0', default=0)
parser.add_argument('--horizon', type=int, help='where hz is a non-negative integer', default=1)
args = parser.parse_args()

rng = default_rng(args.randomSeed)
probabilities = []
N = 0

"""
==================================================================
Generate reward
==================================================================
"""
def gen_reward(arm):
    global probabilities
    reward = rng.choice([0, 1],p=[1-probabilities[arm], probabilities[arm]])
    return reward    

"""
==================================================================
Epsilon-Greedy algorithm
==================================================================
"""
def epsilon_greedy3(epsilon=args.epsilon, horizon=args.horizon):
    global N
    empMeans = np.zeros(N)
    numSamples = np.zeros(N)
    cumulativeReward = 0

    for i in range(horizon):
        randOrHigh = rng.choice([0, 1], p=[epsilon, 1-epsilon]) # if 0 go with rand else high emp mean
        if randOrHigh==0:
            arm = rng.integers(N) # pick arm at random
        else:
            arm = empMeans.argmax() # pick arm with highest empirical mean
        reward = gen_reward(arm)
        empMeans[arm] = (empMeans[arm]*numSamples[arm]+reward)/(numSamples[arm]+1)
        numSamples[arm] += 1
        cumulativeReward += reward

    return cumulativeReward

"""
==================================================================
UCB Algorithm
==================================================================
"""
def ucb(horizon=args.horizon, c=args.scale):
    global N
    empMeans = np.zeros(N)
    numSamples = np.ones(N)
    cumulativeReward = 0

    for t in range(1, horizon+1):
        ucb = empMeans + np.sqrt(c*np.log(t))*np.power(numSamples,-0.5) # get the ucb for this run
        arm = ucb.argmax() # pick arm with highest empirical mean
        reward = gen_reward(arm)
        empMeans[arm] = (empMeans[arm]*numSamples[arm]+reward)/(numSamples[arm]+1)
        numSamples[arm] += 1
        cumulativeReward += reward
    
    return cumulativeReward


"""
==================================================================
Utilities for UCB-KL algorithm
==================================================================
"""
def kl_divergence(x, y):
    if x == 1:
        return x*np.log(x/y)
    elif x == 0:
        return (1-x)*np.log((1-x)/(1-y))
    else:
        return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

def find_q(pa, ua, t, c=3):
    if np.log(t)==0:
        bound = 0
    else:
        bound = (np.log(t) + c*np.log(np.log(t)))/ua
    
    tolerance = 1.0e-3
    start = pa
    end = 1.0
    mid = (start+end)/2
    while np.abs(start-end) > tolerance:
        if kl_divergence(pa, mid) > bound:
            end = mid
        else: 
            start = mid
        mid = (start+end)/2
    q = mid
    return q

"""
==================================================================
UCB-KL Algorithm
==================================================================
"""
def kl_ucb(horizon=args.horizon):
    global N
    empMeans = np.zeros(N)
    numSamples = np.ones(N)
    cumulativeReward = 0

    for t in range(1, horizon+1):
        ucbkl = [find_q(empMeans[i], numSamples[i], t) for i in range(N)]
        arm = np.argmax(ucbkl) # pick arm with highest empirical mean
        reward = gen_reward(arm)
        empMeans[arm] = (empMeans[arm]*numSamples[arm]+reward)/(numSamples[arm]+1)
        numSamples[arm] += 1
        cumulativeReward += reward
    
    return cumulativeReward

"""
==================================================================
Thompson-Sampling
==================================================================
"""
def thompson_sampling():
    global N
    successes = np.zeros(N)
    failures = np.ones(N)
    cumulativeReward = 0

    for t in range(horizon):
        betas = [rng.beta(successes[i]+1, failures[i]+1) for i in range(N)]
        arm = np.argmax(betas) # pick arm with highest empirical mean
        reward = gen_reward(arm)
        if reward==1:
            successes[arm]+=1
        else:
            failures[arm]+=1
        cumulativeReward += reward
    
    return cumulativeReward


if __name__=="__main__":
    algTofuncMap = {
        "epsilon-greedy-t1": epsilon_greedy3,
        "ucb-t1": ucb,
        "kl-ucb-t1": kl_ucb,
        "thompson-sampling-t1": thompson_sampling,
    }

    instance = args.instance
    if instance is None:
        print("NO INSTANCE SELECTED: EXITING...")
        exit(0)

    algorithm = args.algorithm
    if algorithm is None:
        print("NO ALGORITHM SELECTED: EXITING...")
        exit(0)

    randomSeed = args.randomSeed
    epsilon = args.epsilon
    scale = args.scale
    threshold = args.threshold
    horizon = args.horizon

    with open(instance) as file:
        lines = file.readlines()
        probs = [float(line.rstrip()) for line in lines] # Now, probs contains the list of probabilities
    ## SETUP THE GLOBAL VARS
    probabilities = probs
    N = len(probs) 

    REW = algTofuncMap[algorithm]() # REW
    REG = np.max(probs)*args.horizon - REW

    HIGHS = 0 # specific to Task 4

    list_print = [instance, algorithm, randomSeed, epsilon, scale, threshold, horizon, REG, HIGHS]
    print(','.join(str(item) for item in list_print), end='\n')