import argparse
import numpy as np
from pulp import *

parser = argparse.ArgumentParser(description='MDP Instance')
parser.add_argument('--mdp', type=str,
                    help='followed by a path to the input MDP file')
parser.add_argument('--algorithm', type=str,
                    help='followed by one of vi, hpi, and lp.', default='hpi')
args = parser.parse_args()


class MDPPlanner:
    FILE = ""
    TYPES_LIST = {'numStates': int, 'numActions': int,
                  'mdptype': str, 'discount': float}
    MDP_DETAILS = dict()
    MDP_TRANSITIONS = []

    def __init__(self, FILE) -> None:
        self.FILE = FILE

    def parseMDPfile(self):
        with open(self.FILE) as file:
            while (line := file.readline().rstrip()):
                splits = line.split()  # split at whitespaces
                if splits[0] not in ["transition", "end"]:
                    self.MDP_DETAILS[splits[0]] = self.TYPES_LIST[splits[0]](splits[1])
                elif splits[0] == "end":
                    if splits[1] == '-1':
                        self.MDP_DETAILS["end"] = -1
                    else:
                        self.MDP_DETAILS["end"] = [int(x) for x in splits[1:]]
                    # Initialise the initial state of the MDP_TRANSITIONS array
                    self.MDP_TRANSITIONS = [[[] for _ in range(self.MDP_DETAILS["numActions"])] for _ in range(self.MDP_DETAILS["numStates"])]
                elif splits[0] == "transition":
                    self.MDP_TRANSITIONS[int(splits[1])][int(splits[2])].append((int(splits[3]), float(splits[4]), float(splits[5])))


    def valueIteration(self, tol=1e-8):
        ns = self.MDP_DETAILS["numStates"]
        na = self.MDP_DETAILS["numActions"]
        gamma = self.MDP_DETAILS["discount"]
        
        V0 = np.zeros(ns)
        V1 = np.empty_like(V0)
        pi = np.empty_like(V0, dtype=int)
        
        while True:
            for s in range(ns):
                temp = []
                for a in range(na):
                    val = 0
                    for s1,r,p in self.MDP_TRANSITIONS[s][a]:
                        val += p*(r + gamma*V0[s1])
                    temp += [val]
                V1[s] = np.max(temp)
            if np.all(np.abs(V1-V0) < tol):
                break
            np.copyto(V0,V1)
            
        for s in range(ns):
            temp = []
            for a in range(na):
                val = 0
                for s1,r,p in self.MDP_TRANSITIONS[s][a]:
                    val += p*(r + gamma*V1[s1])
                temp += [val]
            pi[s] = np.argmax(temp)

        return pi, V1
    

    def howardsPolicyIteration(self):
        ns = self.MDP_DETAILS["numStates"]
        na = self.MDP_DETAILS["numActions"]
        gamma = self.MDP_DETAILS["discount"]

        pi = np.zeros(ns, dtype=int) # a given policy
        Vpi = np.zeros(ns, dtype=float)
        while True:
            T = np.zeros((ns, ns))
            R = np.zeros((ns, ns))

            for s in range(ns):
                for s1,r,p in self.MDP_TRANSITIONS[s][pi[s]]:
                    T[s][s1] = p
                    R[s][s1] = r 
            b = np.sum(T*R, axis=1)
            A = np.eye(ns) - gamma*T

            Vpi = np.linalg.solve(A, b)
            
            Qpi = np.zeros((ns,na))
            for s in range(ns):
                for a in range(na):
                    for s1,r,p in self.MDP_TRANSITIONS[s][a]:
                        Qpi[s][a] += p*(r + gamma*Vpi[s1])

            loca = np.max(Qpi, axis=1) > Vpi
            if np.all(pi[loca] == np.argmax(Qpi, axis=1)[loca]):
                break
            else:
                pi[loca] = np.argmax(Qpi, axis=1)[loca]
        return pi, Vpi


    def lpFormulation(self):
        ns = self.MDP_DETAILS["numStates"]
        na = self.MDP_DETAILS["numActions"]
        gamma = self.MDP_DETAILS["discount"]

        mdp_planning = LpProblem(name="MDP-Planning", sense=LpMinimize)
        V_lp = LpVariable.dicts("V_lp", range(ns), cat=LpContinuous)
        
        # objective
        mdp_planning += lpSum([V_lp[i] for i in range(ns)]) 
        # constraints
        for s in range(ns):
            for a in range(na):
                mdp_planning += V_lp[s] >= lpSum([p*(r + gamma*V_lp[s1]) for s1,r,p in self.MDP_TRANSITIONS[s][a]])
        mdp_planning.solve(PULP_CBC_CMD(msg=0))

        V = np.zeros(ns, dtype=float)
        for i in range(ns):
            V[i] = V_lp[i].varValue
       
        pi = np.empty_like(V, dtype=int)
        temp = np.zeros((ns,na))
        for s in range(ns):
            for a in range(na):
                for s1,r,p in self.MDP_TRANSITIONS[s][a]:
                    temp[s][a] += p*(r + gamma*V[s1])
        pi = np.argmax(temp, axis=1)

        return pi, V


    def printOutput(self, pi, V):
        for vi, pii in zip(V, pi):
            print("{:.6f}".format(vi), '\t', pii, end='\n')


if __name__ == "__main__":
    mdp_file = args.mdp
    algorithm = args.algorithm
    obj = MDPPlanner(mdp_file)
    obj.parseMDPfile()

    if algorithm == "vi":
        pi, V = obj.valueIteration() # TOLERANCE AT 1e-4 BY DEFAULT
    elif algorithm == "hpi":
        pi, V = obj.howardsPolicyIteration()
    elif algorithm == "lp":
        pi, V = obj.lpFormulation()
        
    obj.printOutput(pi, V)
