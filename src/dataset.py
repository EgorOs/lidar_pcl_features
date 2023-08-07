import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.constants import PROJ_ROOT
from src.features import get_features
from src.utils import apply_class_mapping, remap_class_indexes


class Scene(NamedTuple):
    name: str
    points: NDArray[float]


TRAINING_SET = 'training'
TESTING_SET = 'testing'


class Dataset:
    def __init__(
        self,
        data_folder: Path,
        data_pattern: str = '*.xyz_label_conf',
        cache_dir: Path = PROJ_ROOT / '.dataset_cache',
        drop_cache: bool = False,
    ):
        self.drop_cache = drop_cache
        self.data_folder = data_folder
        self.data_pattern = data_pattern
        self.cache_dir = cache_dir
        self._idx_to_new_idx = None

    def prepare(self):
        if self.drop_cache:
            shutil.rmtree(self.cache_dir)
        all_scenes = list(self.get_scenes(TRAINING_SET)) + list(self.get_scenes(TESTING_SET))
        classes = np.hstack([sc.points[:, 3] for sc in all_scenes])
        self._idx_to_new_idx = remap_class_indexes(classes)

    @lru_cache
    def get_scenes(self, subset: Literal['training', 'testing']) -> Tuple[Scene]:
        folder_iter = (self.data_folder / subset).glob(self.data_pattern)
        return tuple((Scene(path.name, np.genfromtxt(path, delimiter=' ')) for path in folder_iter))

    @lru_cache
    def get_features(self, subset: Literal['training', 'testing']) -> NDArray[float]:
        all_features = []
        subset_cache = self.cache_dir / subset
        os.makedirs(subset_cache, exist_ok=True)
        for scene in tqdm(self.get_scenes(subset), desc='Preparing features for "{0}" scenes:'.format(subset)):
            file_path = subset_cache / f'{scene.name}.npy'
            if file_path.is_file():
                scene_features = np.load(str(file_path))
            else:
                scene_features = get_features(scene.points[:, :3])
                np.save(str(file_path), scene_features)
            all_features.append(scene_features)
        return _post_process_features(np.vstack(all_features))

    @lru_cache
    def get_classes(self, subset: Literal['training', 'testing']) -> NDArray[float]:
        scenes = self.get_scenes(subset)
        classes = np.hstack([sc.points[:, 3] for sc in scenes])
        return apply_class_mapping(classes, self._idx_to_new_idx)


def _post_process_features(features: NDArray[float]) -> NDArray[float]:
    return np.nan_to_num(features, posinf=1e8, neginf=-1e8)
