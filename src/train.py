from pathlib import Path
from typing import Dict

import lightning
import numpy as np
from clearml import OutputModel, Task
from clearml.datasets import Dataset as ClearmlDataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from xgboost import XGBClassifier

from src.annotations import Annotation
from src.config import ExperimentConfig
from src.constants import PROJ_ROOT
from src.dataset import TESTING_SET, TRAINING_SET, Dataset
from src.experiment_tracking import log_point_cloud
from src.utils import read_class_mapping


def train(cfg: ExperimentConfig, idx_to_class: Dict[int, Annotation]):
    lightning.seed_everything(0)
    Task.force_requirements_env_freeze()  # or use task.set_packages() for more granular control.
    task = Task.init(
        project_name=cfg.project_name,
        task_name=cfg.experiment_name,
        # If `output_uri=True` uses default ClearML output URI,
        # can use string value to specify custom storage URI like S3.
        output_uri=True,
    )
    task.connect_configuration(configuration=cfg.model_dump())

    data_path = Path(ClearmlDataset.get(dataset_name=cfg.data_config.dataset_name).get_local_copy())
    dataset = Dataset(data_path, idx_to_class, feature_scales=[3, 6], drop_cache=cfg.data_config.drop_cache)
    dataset.prepare()

    train_features = dataset.get_features(TESTING_SET)  # Test / train are switched intentionally.
    train_classes = dataset.get_classes(TESTING_SET)

    sampler = RandomOverSampler()
    train_features, train_classes = sampler.fit_resample(train_features, train_classes)
    test_features = dataset.get_features(TRAINING_SET)
    test_classes = dataset.get_classes(TRAINING_SET)

    bst = XGBClassifier(verbosity=3, max_depth=6, n_estimators=100)
    classes_weights = list(
        class_weight.compute_class_weight('balanced', classes=np.unique(train_classes), y=train_classes),
    )
    bst.fit(train_features, train_classes, sample_weight=[classes_weights[cls] for cls in train_classes])
    preds = bst.predict(test_features)

    cf_matrix = confusion_matrix(test_classes, preds)
    cls_report = classification_report(test_classes, preds)
    cf_labels = [dataset.idx_to_class_name[idx] for idx, _ in enumerate(cf_matrix)]
    task.logger.report_confusion_matrix(
        'Confusion matrix',
        'ignored',
        iteration=None,
        matrix=cf_matrix,
        xaxis=None,
        yaxis=None,
        xlabels=cf_labels,
        ylabels=cf_labels,
    )
    task.logger.report_text(cls_report)

    vis_scene = 'oakland_part3_an_training.xyz_label_conf'
    vis_scene_points = dataset.get_scene_points(TRAINING_SET, vis_scene)
    vis_scene_features = dataset.get_scene_features(TRAINING_SET, vis_scene)
    vis_scene_classes = dataset.get_scene_classes(TRAINING_SET, vis_scene)
    vis_pred_classes = bst.predict(vis_scene_features)

    log_point_cloud(task, dataset, vis_scene_points, vis_pred_classes, 'Visualized predictions')
    log_point_cloud(task, dataset, vis_scene_points, vis_scene_classes, 'Visualized ground truth')

    output_model = OutputModel(
        task=task,
        label_enumeration={name: idx for idx, name in dataset.idx_to_class_name.items()},
    )
    checkpoint_name = 'model.json'
    bst.save_model(checkpoint_name)
    output_model.update_weights(checkpoint_name)


if __name__ == '__main__':
    cfg_path = PROJ_ROOT / 'configs'
    train(
        cfg=ExperimentConfig.from_yaml(cfg_path / 'train.yaml'),
        idx_to_class=(read_class_mapping(cfg_path / 'class_mapping.yaml')),
    )
