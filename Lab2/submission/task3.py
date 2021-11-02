import os
import shutil
import subprocess

STATE_P1 = "data/attt/states/states_file_p1.txt"
STATE_P2 = "data/attt/states/states_file_p2.txt"

# DIALS to change ############################
# Initial policy that Player 2 starts with   #
POL0 = "data/attt/policies/p2_policy2.txt"   #
shutil.copy(POL0, "pol0")                    #
OPP0 = 2                                     #
RUNS = 20                                    #
##############################################

## INIT ######################################
policy = POL0                                #
states = STATE_P1 if OPP0==2 else STATE_P2   #
player = 1 if OPP0==2 else 2                 #
ABSOLUTE_DIFF = {1: [], 2: []}               #
##############################################
for i in range(1,RUNS+1):
    cmd_encoder = "python","encoder.py","--policy",policy,"--states",states
    f = open(f'mdp{i}','w')
    subprocess.run(cmd_encoder,stdout=f)
    f.close()

    cmd_planner = "python","planner.py","--mdp",f"mdp{i}"
    f = open(f'planner{i}','w')
    subprocess.run(cmd_planner,stdout=f)
    f.close()

    cmd_decoder = "python","decoder.py","--value-policy",f"planner{i}","--states",states,"--player-id",str(player)
    f = open(f'pol{i}','w')
    subprocess.run(cmd_decoder,stdout=f)
    f.close()

    pol_old = []
    pol_new = []
    if i>=3:
        with open(f'pol{i-2}', 'r') as file_old, open(f'pol{i}', 'r') as file_new:
            pl = int(file_old.readline().rstrip())
            file_new.readline()
            while (line := file_old.readline().rstrip()):
                splits = line.split()
                for idx, spl in enumerate(splits[1:]):
                    if spl=='1':
                        pol_old+=[idx]
                        break
            while (line := file_new.readline().rstrip()):
                splits = line.split()
                for idx, spl in enumerate(splits[1:]):
                    if spl=='1':
                        pol_new+=[idx]
                        break
        diff = 0
        for s1, s2 in zip(pol_old, pol_new):
            diff+=(s1!=s2)
        ABSOLUTE_DIFF[pl]+=[diff]

    player = 1 if player==2 else 2                 
    policy = f'pol{i}'
    states = STATE_P1 if player==1 else STATE_P2

    os.remove(f"mdp{i}")
    os.remove(f"planner{i}")

for i in [1,2]:
    print(f"Player {i} - Mismatches with previous:",*ABSOLUTE_DIFF[i])

for i in range(RUNS+1):
    os.remove(f"pol{i}")
    