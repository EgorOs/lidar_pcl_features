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
