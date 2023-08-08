from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import distinctipy
import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf


@dataclass
class Annotation:
    class_name: str
    color: tuple


def read_class_mapping(mapping_path: Path) -> Dict[int, Annotation]:
    cfg = OmegaConf.to_container(OmegaConf.load(mapping_path), resolve=True)
    colors = distinctipy.get_colors(len(cfg['idx_to_class']))
    return {
        int(idx): Annotation(cls_name, colors.pop(0)) for idx, cls_name in cfg['idx_to_class'].items()  # noqa: WPS221
    }


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
