# Reproducible Inference: DimeNet++ on QM9
**Student:** Adesh Srivastava

This repository contains a fully reproducible inference workflow for evaluating the **DimeNet++** Graph Neural Network on the **QM9** benchmark dataset, executed on the Georgia Tech AI Makerspace (PACE ICE) cluster.

## 1. Environment & Setup
The software environment is managed via Conda to ensure PyTorch and PyTorch Geometric correctly interface with the cluster's CUDA 11.8 drivers.

**Setup Instructions:**
1. Clone this repository and navigate into it.
2. Create the environment from the provided YAML file:
   `conda env create -f environment.yml`
3. Activate the environment:
   `conda activate dimenet-env`

## 2. Artifact Traceability (Model, Data, & Provenance)
* **Model Architecture:** DimeNet++ (Directional Message Passing Neural Network). Implemented natively via `torch_geometric.nn.models.DimeNetPlusPlus`.
* **Dataset:** QM9 Benchmark. Contains 130k small organic molecules. Automatically downloaded via `torch_geometric.datasets.QM9` from the MoleculeNet DeepChem mirror.
* **Primary Paper:** Gasteiger et al., *Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules* (NeurIPS 2020). Source repository: `https://github.com/gasteigerjo/dimenet`.

## 3. Inference Workflow & Job Scripts
The inference pipeline is contained in `evaluate.py`. It constructs the 3D spatial graphs (atomic numbers and coordinates) and passes them through the DimeNet++ interaction blocks to predict physical properties.

**To run on AI Makerspace (SLURM):**
Submit the job script directly to the scheduler:
`sbatch run_inference.slurm`

The script requests 1 standard GPU, 16GB of RAM, and 4 CPU cores, fitting well within the 32 GPU-hour PACE ICE allocation limits.

## 4. Validation of Results
This workflow includes two levels of validation, directly embedded in the execution logs:
1. **Smoke Test:** Executes `python evaluate.py --smoke-test`. This runs a single batch of 32 molecules to verify that the environment is functional, the dataset downloads correctly, and the model tensors align without CUDA memory crashes.
2. **Substantive Validation:** Executes a full evaluation over the 10,000-molecule test split targeting index 7 (Internal Energy $U_0$ at 0K). It calculates the Mean Absolute Error (MAE), verifying the directional message-passing execution across the dataset.

## 5. Execution Evidence
Execution evidence is provided in the repository as `execution_*.log`. This file contains the node and GPU allocation data, dataset processing milestones, expected output tensor shapes from the smoke test, and the final MAE metric from the substantive validation run.