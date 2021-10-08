import argparse

parser = argparse.ArgumentParser(description='MDP Instance')
parser.add_argument('--policy', type=str,
                    help='followed by a path to the policy file')
parser.add_argument('--states', type=str,
                    help='followed by a path to the state file')
args = parser.parse_args()

class Encoder:
    states = dict() # str -> int
    policies = dict() # str -> []

    def __init__(self, policy, states) -> None:
        self.policyFile = policy
        self.statesFile = states
    

    def parseFiles(self):
        with open(self.statesFile) as sf:
            k = 0
            while (line := sf.readline().rstrip()):
                self.states[line] = k
                k += 1
        
        with open(self.policyFile) as pf:
            _ = pf.readline()
            while (line := pf.readline().rstrip()):
                splits = line.split()
                self.policies[splits[0]] = [float(x) for x in splits[1:]]


    def formulateMDP(self):
        numActions = 9
        numStates = len(self.states) + 1
        discount = 1
        end = numStates - 1 # mark last state as terminal state
        
        def isTerminal(state, pl=1):
            st = str(pl)*3
            row = state[0:3]==st or state[3:6]==st or state[6:9]==st
            col = state[0::3]==st or state[1::3]==st or state[2::3]==st
            diag = state[::4]==st or state[2:7:2]==st
            return row or col or diag

        MDP_TRANSITIONS = [[[] for _ in range(numActions)] for _ in range(numStates)]

        for state, idx in self.states.items():
            for ac in range(numActions):
                if state[ac] != '0':
                    continue
                nextstate = state[:ac] + '1' + state[(ac+1):]

                if isTerminal(nextstate, pl=1) or ('0' not in nextstate):
                    MDP_TRANSITIONS[idx][ac].append((end, 0.0, 1.0))
                    continue

                for ac2, ac2_prob in enumerate(self.policies[nextstate]):
                    if ac2_prob == 0:
                        continue
                    ns2 = nextstate[:ac2] + '2' + nextstate[(ac2+1):]
                    
                    if isTerminal(ns2, pl=2):
                        MDP_TRANSITIONS[idx][ac].append((end, 1.0, ac2_prob))
                    elif '0' not in ns2:
                        MDP_TRANSITIONS[idx][ac].append((end, 0.0, ac2_prob))
                    else:
                        MDP_TRANSITIONS[idx][ac].append((self.states[ns2], 0.0, ac2_prob))

        # Print stuff
        print("numStates", numStates)
        print("numActions", numActions)
        print("end", end)
        for s in range(numStates):
            for a in range(numActions):
                for s1,r,p in MDP_TRANSITIONS[s][a]:
                    print("transition", s, a, s1, r, p)
        print("mdptype", "episodic")
        print("discount", discount)


if __name__=="__main__":
    # python3 encoder.py --policy pa2_base/data/attt/policies/p2_policy1.txt --states pa2_base/data/attt/states/states_file_p1.txt
    policyFile = args.policy
    statesFile = args.states

    obj = Encoder(policyFile, statesFile)
    obj.parseFiles()
    obj.formulateMDP()