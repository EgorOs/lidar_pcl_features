from clearml import Task
from numpy.typing import NDArray
from plotly import graph_objects as go

from src.dataset import Dataset


def log_point_cloud(task: Task, dataset: Dataset, pts3d: NDArray[float], pt_classes: NDArray[float], plot_name: str):
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
    task.get_logger().report_plotly(title=plot_name, series=plot_name, iteration=0, figure=fig)
