#!/bin/bash

# TASK 1
instances=("instances/instances-task1/i-1.txt" "instances/instances-task1/i-2.txt" "instances/instances-task1/i-3.txt")
algorithms=("epsilon-greedy-t1" "ucb-t1" "kl-ucb-t1" "thompson-sampling-t1")
horizons=(100 400 1600 6400 25600 102400)
randomSeeds=( $(seq 0 49 ) )

for in in "${instances[@]}"; do
    for al in "${algorithms[@]}"; do
        for h in "${horizons[@]}"; do
            for rs in "${randomSeeds[@]}"; do
                python3 bandit.py --instance "$in" --algorithm "$al" --randomSeed "$rs" --epsilon 0.02 --scale 2 --threshold 0 --horizon "$h" >> outputData.txt
            done
        done
    done
done

# TASK 2
# algorithm set to ucb-t2
# horizon equal to 10000
instances=("instances/instances-task2/i-1.txt" "instances/instances-task2/i-2.txt" "instances/instances-task2/i-3.txt" "instances/instances-task2/i-4.txt", "instances/instances-task2/i-5.txt")
scales=( $(seq 0.02 0.02 0.3) )
randomSeeds=( $(seq 0 49 ) )

for in in "${instances[@]}"; do
    for sl in "${scales[@]}"; do
        for rs in "${randomSeeds[@]}"; do
            python3 bandit.py --instance "$in" --algorithm ucb-t2 --randomSeed "$rs" --epsilon 0.02 --scale "$sl" --threshold 0 --horizon 10000 >> outputData.txt
        done
    done
done

# TASK 3
# algorithm set to alg-t3
instances=("instances/instances-task3/i-1.txt" "instances/instances-task3/i-2.txt")
horizons=(100 400 1600 6400 25600 102400)
randomSeeds=( $(seq 0 49) )

for in in "${instances[@]}"; do
    for h in "${horizons[@]}"; do
        for rs in "${randomSeeds[@]}"; do
            python3 bandit.py --instance "$in" --algorithm alg-t3 --randomSeed "$rs" --epsilon 0.02 --scale 2 --threshold 0 --horizon "$h" >> outputData.txt
        done
    done
done

# TASK 4
# algorithm set to alg-t4
instances=("instances/instances-task4/i-1.txt" "instances/instances-task4/i-2.txt")
thresholds=(0.2 0.6)
horizons=(100 400 1600 6400 25600 102400)
randomSeeds=( $(seq 0 49) )

for in in "${instances[@]}"; do
    for h in "${horizons[@]}"; do
        for rs in "${randomSeeds[@]}"; do
            python3 bandit.py --instance "$in" --algorithm alg-t4 --randomSeed "$rs" --epsilon 0.02 --scale 2 --threshold 0 --horizon "$h" >> outputData.txt
        done
    done
done