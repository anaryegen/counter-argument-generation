#!/bin/bash
#SBATCH --job-name=prometheus
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

export HF_HUB_CACHE=/ncache/hub
export TRANSFORMERS_CACHE=/ncache/hub
#export HF_HOME=.cache/huggingface/hub/
python3 eval.py
