# lidar_pcl_features

LiDAR point cloud segmentation with classic ML approaches.

<a href="https://clear.ml/docs/latest/"><img alt="Config: Hydra" src="https://img.shields.io/badge/MLOps-Clear%7CML-%2309173c"></a>

# Getting started

1. Follow [instructions](https://github.com/python-poetry/install.python-poetry.org)
   to install Poetry:
   ```bash
   # Unix/MacOs installation
   curl -sSL https://install.python-poetry.org | python3 -
   ```
1. Check that poetry was installed successfully:
   ```bash
   poetry --version
   ```
1. Setup workspace:
   ```bash
   make setup_ws
   ```
1. Setup ClearML:
   ```bash
   clearml-init
   ```
1. Migrate dataset to your ClearML workspace:
   ```bash
   make migrate_dataset
   ```
1. (Optional) Configure and run Jupyter lab:
   ```bash
   make jupyterlab_start
   ```

# Train

```bash
make run_training
```

# Results

🧪 [Experiment link](https://app.clear.ml/projects/27c1eb5330ef46d8a0e021fef065689e/experiments/1b7beb5bfdd84599b9e37d73d9dd58d7/output/execution)

<img src=assets/results_1.png>
<img src=assets/results_2.png>

```bash
              precision    recall  f1-score
           0       0.52      0.73      0.61
           1       0.99      0.85      0.91
           2       0.35      0.50      0.41
           3       0.89      0.94      0.91
           4       0.45      0.25      0.32
    accuracy                           0.82
   macro avg       0.64      0.65      0.63
weighted avg       0.83      0.82      0.82
```
