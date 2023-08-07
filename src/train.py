import os
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d
from clearml.datasets import Dataset as ClearmlDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.annotations import Annotation
from src.config import ExperimentConfig
from src.features import LocalPCD
from src.utils import adjust_idx_to_class_mapping, read_class_mapping

PROJ_ROOT = Path(os.getenv('PROJ_ROOT', Path(__file__).resolve().parents[1]))


# class Workflow(lt.LightningWork):
#     def run(self, *args: Any, **kwargs: Any) -> None:
#         pass


def train(cfg: ExperimentConfig, idx_to_class: Dict[int, Annotation]):
    data_path = Path(ClearmlDataset.get(dataset_name=cfg.data_config.dataset_name).get_local_copy())

    sample_path = data_path / 'testing' / 'oakland_part2_ac.xyz_label_conf'

    raw_points = np.genfromtxt(sample_path, delimiter=' ')
    # rgb_points = colorize_points(raw_points, idx_to_class)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points[:, :3])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    training_features = []
    for pt3d in pcd.points:
        kn, idx, *_ = pcd_tree.search_radius_vector_3d(pt3d, 3)
        local_pcd = LocalPCD.from_pt_and_neighbours(pcd, pt3d, idx)
        training_features.append(local_pcd.features)

    training_classes, idx_to_class = adjust_idx_to_class_mapping(raw_points[:, 3].astype(np.int32), idx_to_class)
    X_train, X_test, y_train, y_test = train_test_split(np.vstack(training_features), training_classes, test_size=0.2)

    bst = XGBClassifier(verbosity=3)
    bst.fit(X_train, y_train)
    preds = bst.predict(X_test)
    print(f'preds: {preds}')

    cf_matrix = confusion_matrix(y_test, preds)
    cls_report = classification_report(y_test, preds)


if __name__ == '__main__':
    cfg_path = PROJ_ROOT / 'configs'
    train(
        cfg=ExperimentConfig.from_yaml(cfg_path / 'train.yaml'),
        idx_to_class=(read_class_mapping(cfg_path / 'class_mapping.yaml')),
    )
