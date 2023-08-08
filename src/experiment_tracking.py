from typing import Optional

from clearml import Task
from numpy.typing import NDArray
from plotly import graph_objects as go

from src.dataset import Dataset


def log_point_cloud(
    task: Task,
    dataset: Dataset,
    pts3d: NDArray[float],
    pt_classes: NDArray[float],
    title: str,
    series: Optional[str] = None,
):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts3d[pt_classes == idx][:, 0],
                y=pts3d[pt_classes == idx][:, 1],
                z=pts3d[pt_classes == idx][:, 2],
                mode='markers',
                marker={
                    'size': 1,
                    'color': '#eeeeee',  # Setting solid color results in unique colors in ClearML UI.
                },
                name=class_name,
            )
            for idx, class_name in dataset.idx_to_class_name.items()
        ],
    )
    series = title if series is None else series
    task.get_logger().report_plotly(title=title, series=series, iteration=0, figure=fig)


def visualize_scene(task: Task, model, dataset: Dataset, subset, scene_name: str):
    points3d = dataset.get_scene_points(subset, scene_name)
    features = dataset.get_scene_features(subset, scene_name)
    gt_classes = dataset.get_scene_classes(subset, scene_name)
    preds = model.predict(features)
    log_point_cloud(task, dataset, points3d, gt_classes, f'{subset} / {scene_name}', 'Ground truth')
    log_point_cloud(task, dataset, points3d, preds, f'{subset} / {scene_name}', 'Predictions')
