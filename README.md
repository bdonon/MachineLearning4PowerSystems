# Main Repo
[Repository of our paper Topology-Aware Reinforcement Learning for Tertiary Voltage Control presented at PSCC 2024.](https://github.com/bdonon/PSCC2024)

# Installation
```
git clone https://github.com/bdonon/ml4ps
cd ml4ps
git checkout -b pscc2024
```

# Data
Links to the three generated datasets are provided below. Notice that it also contains the test sets modified by the baseline and the trained policies.

  - [*Standard* dataset](https://zenodo.org/doi/10.5281/zenodo.8367764)

  - [*Condenser* dataset](https://zenodo.org/doi/10.5281/zenodo.8367613)

  - [*Reduced* dataset](https://zenodo.org/doi/10.5281/zenodo.8367756)

# Training and Testing
Experiment on a given dataset.
```
python main.py data_dir=PATH_TO_TRAIN_DATASET eval_data_dir=PATH_TO_EVAL_DATASET test_data_dir=PATH_TO_TEST_DATASET eval_size=2000 test_size=10000
```