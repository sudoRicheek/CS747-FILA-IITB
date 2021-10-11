#!/bin/bash

python encoder.py --policy data/attt/policies/p1_policy1.txt --states data/attt/states/states_file_p2.txt > mymdp_p1_policy1_states_file_p2.txt
python encoder.py --policy data/attt/policies/p1_policy2.txt --states data/attt/states/states_file_p2.txt > mymdp_p1_policy2_states_file_p2.txt
python encoder.py --policy data/attt/policies/p2_policy1.txt --states data/attt/states/states_file_p1.txt > mymdp_p2_policy1_states_file_p1.txt
python encoder.py --policy data/attt/policies/p2_policy2.txt --states data/attt/states/states_file_p1.txt > mymdp_p2_policy2_states_file_p1.txt

python planner.py --mdp mymdp_p1_policy1_states_file_p2.txt > myvp_p1_policy1_states_file_p2.txt
python planner.py --mdp mymdp_p1_policy2_states_file_p2.txt > myvp_p1_policy2_states_file_p2.txt
python planner.py --mdp mymdp_p2_policy1_states_file_p1.txt > myvp_p2_policy1_states_file_p1.txt
python planner.py --mdp mymdp_p2_policy2_states_file_p1.txt > myvp_p2_policy2_states_file_p1.txt

python decoder.py --value-policy myvp_p1_policy1_states_file_p2.txt --states data/attt/states/states_file_p2.txt --player-id 2 > mypolicy_p1_policy1_states_file_p2.txt
python decoder.py --value-policy myvp_p1_policy2_states_file_p2.txt --states data/attt/states/states_file_p2.txt --player-id 2 > mypolicy_p1_policy2_states_file_p2.txt
python decoder.py --value-policy myvp_p2_policy1_states_file_p1.txt --states data/attt/states/states_file_p1.txt --player-id 1 > mypolicy_p2_policy1_states_file_p1.txt
python decoder.py --value-policy myvp_p2_policy2_states_file_p1.txt --states data/attt/states/states_file_p1.txt --player-id 1 > mypolicy_p2_policy2_states_file_p1.txt

echo -e "Against player 1 policy 1, losses:"

j=0
for i in {1..500}
do
   str="$(python attt.py -p1 data/attt/policies/p1_policy1.txt -p2 mypolicy_p1_policy1_states_file_p2.txt -rs $i | tail -1)"
   j=0
   if [ "${str:7:1}" != "2" ]; then
     j=$(( $j + 1 ))
   fi
done
echo $j

echo -e "Against player 1 policy 2, losses:"

j=0
for i in {1..500}
do
   str="$(python attt.py -p1 data/attt/policies/p1_policy2.txt -p2 mypolicy_p1_policy2_states_file_p2.txt -rs $i | tail -1)"
   if [ "${str:7:1}" != "2" ]; then
     j=$(( $j + 1 ))
   fi
done
echo $j

echo -e "Against player 2 policy 1, losses:"

j=0
for i in {1..500}
do
   str="$(python attt.py -p1 mypolicy_p2_policy1_states_file_p1.txt -p2 data/attt/policies/p2_policy1.txt -rs $i | tail -1)"
   if [ "${str:7:1}" != "1" ]; then
     j=$(( $j + 1 ))
   fi
done
echo $j

echo -e "Against player 2 policy 2, losses:"

j=0
for i in {1..500}
do
   str="$(python attt.py -p1 mypolicy_p2_policy2_states_file_p1.txt -p2 data/attt/policies/p2_policy2.txt -rs $i | tail -1)"
   if [ "${str:7:1}" != "1" ]; then
     j=$(( $j + 1 ))
   fi
done
echo $j