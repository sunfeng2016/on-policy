#!/bin/sh
env="SCE"
scenario="defense"
map="100_vs_100"
algo="rmappo"
exp="eval"
seed_max=1

# run_cmd="python"
run_cmd="python -m debugpy --listen 8888 --wait-for-client"

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 ${run_cmd} ../train/train_sce.py --env_name ${env} --algorithm_name ${algo} \ --scenario_name ${scenario} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 100000 --ppo_epoch 10 --use_value_active_masks --use_eval --eval_episodes 32 \
    --only_eval True --model_dir '/home/ubuntu/sunfeng/MARL/on-policy/onpolicy/scripts/results/SCE/defense/100_vs_100/rmappo/check/wandb/run-20240615_231926-cbd5yhu4/files'
done
