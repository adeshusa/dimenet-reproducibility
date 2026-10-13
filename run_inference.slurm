#!/bin/bash
#SBATCH --job-name=dimenet_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1               
#SBATCH --time=00:30:00            
#SBATCH --output=execution_%j.log  

echo "Starting DimeNet++ Job on PACE ICE"
echo "Node: $HOSTNAME"
echo "GPU Allocated: $CUDA_VISIBLE_DEVICES"

# Load necessary modules for GT PACE ICE
module load anaconda3

# Activate the conda environment
source activate dimenet-env

# 1. Run the Smoke Test Validation
echo "--- RUNNING SMOKE TEST ---"
python evaluate.py --smoke-test

# 2. Run the Full Inference Validation
echo "--- RUNNING FULL INFERENCE ---"
python evaluate.py --target 7

echo "Job completed successfully."