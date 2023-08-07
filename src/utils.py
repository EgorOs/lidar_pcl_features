from pathlib import Path
from typing import Dict, Tuple

import distinctipy
import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.annotations import Annotation


def read_class_mapping(mapping_path: Path) -> Dict[int, Annotation]:
    cfg = OmegaConf.to_container(OmegaConf.load(mapping_path), resolve=True)
    colors = distinctipy.get_colors(len(cfg['idx_to_class']))
    return {int(idx): Annotation(cls, colors.pop(0)) for idx, cls in cfg['idx_to_class'].items()}


def colorize_points(points_with_cls: NDArray[float], idx_to_cls: Dict[str, Annotation]) -> NDArray[float]:
    class_idx_arr = points_with_cls[:, 3]
    idx_to_color = {idx: ann.color for idx, ann in idx_to_cls.items()}
    np.fromfunction(lambda i: idx_to_color[class_idx_arr[i]], shape=class_idx_arr.shape)


def adjust_idx_to_class_mapping(
    classes: NDArray[int],
    idx_to_cls: Dict[int, Annotation],
) -> Tuple[NDArray[int], Dict[int, Annotation]]:
    new_idx_to_class = {}
    idx_to_new_idx = {}
    ctr = 0
    encoded = []
    for class_idx in classes:
        if class_idx not in idx_to_new_idx:
            class_name = idx_to_cls[class_idx]
            idx_to_new_idx[class_idx] = ctr
            new_idx_to_class[ctr] = class_name
            ctr += 1
        encoded.append(idx_to_new_idx[class_idx])
    return np.array(encoded), new_idx_to_class


def remap_class_indexes(classes: NDArray[int]) -> Dict[int, int]:
    idx_to_new_idx = {}
    ctr = 0
    for class_idx in classes:
        if class_idx not in idx_to_new_idx:
            idx_to_new_idx[class_idx] = ctr
            ctr += 1
    return idx_to_new_idx


def apply_class_mapping(classes: NDArray[int], idx_to_new_idx: Dict[int, int]) -> NDArray[int]:
    encoded = []
    for class_idx in classes:
        encoded.append(idx_to_new_idx[class_idx])
    return np.array(encoded)
