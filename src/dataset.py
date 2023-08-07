import os
import shutil
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Dict, Literal, NamedTuple, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.annotations import Annotation
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
        idx_to_cls: Dict[int, Annotation],
        data_pattern: str = '*.xyz_label_conf',
        cache_dir: Path = PROJ_ROOT / '.dataset_cache',
        drop_cache: bool = False,
    ):
        self.data_folder = data_folder
        self.idx_to_cls = idx_to_cls
        self.drop_cache = drop_cache
        self.data_pattern = data_pattern
        self.cache_dir = cache_dir
        self._idx_to_new_idx = None

    def prepare(self):
        if self.drop_cache:
            shutil.rmtree(self.cache_dir)
        all_scenes = list(self.get_scenes(TRAINING_SET)) + list(self.get_scenes(TESTING_SET))
        classes = np.hstack([sc.points[:, 3] for sc in all_scenes]).astype(np.int32)
        self._idx_to_new_idx = remap_class_indexes(classes)

    @cached_property
    def idx_to_class_name(self) -> Dict[int, str]:
        return {
            self._idx_to_new_idx[idx]: ann.class_name
            for idx, ann in self.idx_to_cls.items()
            if idx in self._idx_to_new_idx
        }

    @cached_property
    def idx_to_color(self) -> Dict[int, str]:
        return {
            self._idx_to_new_idx[idx]: ann.color for idx, ann in self.idx_to_cls.items() if idx in self._idx_to_new_idx
        }

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
            scene_features = self.get_scene_features(subset, scene)
            all_features.append(scene_features)
        return _post_process_features(np.vstack(all_features))

    def get_scene_features(self, subset: Literal['training', 'testing'], scene: Union[Scene, str]):
        if isinstance(scene, str):
            scene = self._get_scene_by_name(scene, subset)
        subset_cache = self.cache_dir / subset
        file_path = subset_cache / f'{scene.name}.npy'
        if file_path.is_file():
            scene_features = np.load(str(file_path))
        else:
            scene_features = get_features(scene.points[:, :3])
            np.save(str(file_path), scene_features)
        return scene_features

    @lru_cache
    def get_classes(self, subset: Literal['training', 'testing']) -> NDArray[float]:
        scenes = self.get_scenes(subset)
        classes = np.hstack([sc.points[:, 3] for sc in scenes])
        return apply_class_mapping(classes, self._idx_to_new_idx)

    def get_scene_classes(self, subset: Literal['training', 'testing'], scene_name: str) -> NDArray[float]:
        scene = self._get_scene_by_name(scene_name, subset)
        classes = np.hstack([scene.points[:, 3]])
        return apply_class_mapping(classes, self._idx_to_new_idx)

    def _get_scene_by_name(self, scene_name, subset):
        scene = next((sc for sc in self.get_scenes(subset) if sc.name == scene_name), None)
        if scene is None:
            raise ValueError('Could not find scene {0}'.format(scene_name))
        return scene

    def get_scene_points(self, subset: Literal['training', 'testing'], scene_name: str) -> NDArray[float]:
        scene = self._get_scene_by_name(scene_name, subset)
        return scene.points


def _post_process_features(features: NDArray[float]) -> NDArray[float]:
    features = np.nan_to_num(features, posinf=1e8, neginf=-1e8)
    return np.clip(features, -100, 100)
