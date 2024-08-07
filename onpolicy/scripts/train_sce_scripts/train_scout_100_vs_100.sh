#!/bin/sh
env="SCE"
scenario="scout"
map="100_vs_100"
algo="rmappo"
exp="train_v8"
# exp="debug"
seed_max=1

run_cmd="python"
# run_cmd="python -m debugpy --listen 8888 --wait-for-client"

source /home/ubuntu/.conda/envs/sunfeng/bin/activate pymarl

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 ${run_cmd} ../train/train_sce.py --env_name ${env} --scenario_name ${scenario} \
    --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 --episode_length 600 \
    --num_env_steps 10000000 --ppo_epoch 10 --use_value_active_masks --use_eval --eval_episodes 32 --use_mix_critic True 
done
