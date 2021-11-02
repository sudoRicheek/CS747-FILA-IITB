import argparse

parser = argparse.ArgumentParser(description='MDP Instance')
parser.add_argument('--value-policy', type=str,
                    help='followed by a path to the value_and_policy file')
parser.add_argument('--states', type=str,
                    help='followed by a path to the state file')
parser.add_argument('--player-id', type=str,
                    help='followed by playeri')
args = parser.parse_args()

class Decoder:
    states = []
    indexActions = []
    
    def __init__(self, vpfile, statesfile, playeri) -> None:
        self.vpFile = vpfile
        self.statesFile = statesfile
        self.playeri = playeri


    def parseFiles(self):
        with open(self.statesFile) as sf:
            while (line := sf.readline().rstrip()):
                self.states += [line]
        
        with open(self.vpFile) as vpf:
            while (line := vpf.readline().rstrip()):
                splits = line.split()
                self.indexActions += [int(splits[1])]        
        self.indexActions = self.indexActions[:len(self.states)]


    def formulatePolicy(self):
        numActions = 9
        numStates = len(self.states)

        policy = [[0 for _ in range(numActions)] for _ in range(numStates)]
        for idx, ac in enumerate(self.indexActions):
            policy[idx][ac] = 1
        
        print(self.playeri)
        for idx, pol in enumerate(policy):
            print(self.states[idx], *pol)


if __name__=="__main__":
    vpfile = args.value_policy
    statesfile = args.states
    playeri = args.player_id

    obj = Decoder(vpfile, statesfile, playeri)
    obj.parseFiles()
    obj.formulatePolicy()
    