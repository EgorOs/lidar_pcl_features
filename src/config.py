from pathlib import Path
from typing import Tuple, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset_name: str = 'lidar_cvpr09'
    data_split: Tuple[float, ...] = (0.7, 0.2, 0.1)
    drop_cache: bool = False


class ExperimentConfig(BaseModel):
    project_name: str = 'lidar_pcl_features'
    experiment_name: str = 'lidar_xgboost'
    data_config: DataConfig = Field(default=DataConfig())

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]):
        with open(path, 'w') as out_file:
            yaml.safe_dump(self.dict(), out_file, default_flow_style=False, sort_keys=False)
