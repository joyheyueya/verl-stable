eval "$(conda shell.bash hook)"
conda activate verl

export CUDA_VISIBLE_DEVICES=7
python /home/ubuntu/verl-stable/verl/utils/reward_score/insight/api_server.py