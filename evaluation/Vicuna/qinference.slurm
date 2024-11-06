#!/bin/bash
#SBATCH --job-name=whatever_eval
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

# export HF_HUB_CACHE=/ncache/hub
# export TRANSFORMERS_CACHE=/ncache/hub

python3 vicuna_eval_scores.py
