from pathlib import Path
from typing import Dict

from clearml.datasets import Dataset as ClearmlDataset
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from src.annotations import Annotation
from src.config import ExperimentConfig
from src.constants import PROJ_ROOT
from src.dataset import TESTING_SET, TRAINING_SET, Dataset
from src.utils import read_class_mapping

# class Workflow(lt.LightningWork):
#     def run(self, *args: Any, **kwargs: Any) -> None:
#         pass


def train(cfg: ExperimentConfig, idx_to_class: Dict[int, Annotation]):
    data_path = Path(ClearmlDataset.get(dataset_name=cfg.data_config.dataset_name).get_local_copy())
    dataset = Dataset(data_path, drop_cache=cfg.data_config.drop_cache)
    dataset.prepare()

    train_features = dataset.get_features(TESTING_SET)  # Test / train are switched intentionally.
    train_classes = dataset.get_classes(TESTING_SET)

    test_features = dataset.get_features(TRAINING_SET)
    test_classes = dataset.get_classes(TRAINING_SET)

    bst = XGBClassifier(verbosity=3)
    bst.fit(train_features, train_classes)
    preds = bst.predict(test_features)
    print(f'preds: {preds}')

    cf_matrix = confusion_matrix(test_classes, preds)
    cls_report = classification_report(test_classes, preds)


if __name__ == '__main__':
    cfg_path = PROJ_ROOT / 'configs'
    train(
        cfg=ExperimentConfig.from_yaml(cfg_path / 'train.yaml'),
        idx_to_class=(read_class_mapping(cfg_path / 'class_mapping.yaml')),
    )
