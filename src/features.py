import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from tqdm import tqdm


@dataclass
class LocalPCD:  # noqa: WPS214
    pcd: o3d.geometry.PointCloud = field(default_factory=o3d.geometry.PointCloud)

    @classmethod
    def from_pt_and_neighbours(cls, pcd: o3d.geometry.PointCloud, pt: NDArray[float], nbr_idxs) -> 'LocalPCD':
        nbrs = np.asarray(pcd.points)[nbr_idxs[1:], :]
        local_pcd = o3d.geometry.PointCloud()
        points = np.vstack([pt, nbrs])
        local_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        return cls(local_pcd)

    @cached_property
    def cov_eigvals(self) -> NDArray[float]:
        cov = self.pcd.compute_mean_and_covariance()[1]
        # Expected to have e1 >= e2 >= e3
        return np.array(sorted(np.linalg.eig(cov)[0], reverse=True))

    @cached_property
    def feat_linearity(self) -> float:
        e1, e2, e3 = self.cov_eigvals
        return (e1 - e2) / e1

    @cached_property
    def feat_planarity(self) -> float:
        e1, e2, e3 = self.cov_eigvals
        return (e2 - e3) / e1

    @cached_property
    def feat_scattering(self) -> float:
        e1, e2, e3 = self.cov_eigvals
        return e3 / e1

    @cached_property
    def feat_omnivariance(self) -> float:
        e1, e2, e3 = self.cov_eigvals
        return math.pow((e1 * e2 * e3), 1 / 3)

    @cached_property
    def feat_anisotropy(self) -> float:
        e1, e2, e3 = self.cov_eigvals
        return (e1 - e3) / e1

    @cached_property
    def feat_eigentropy(self) -> float:
        eigen_vals = self.cov_eigvals
        return -np.sum(eigen_vals * np.log(eigen_vals))

    @cached_property
    def feat_eigensum(self) -> float:
        return np.sum(self.cov_eigvals)

    @cached_property
    def feat_change_in_curvature(self) -> float:
        e1, e2, e3 = self.cov_eigvals
        return e3 / (e1 + e2 + e3)

    @cached_property
    def features(self) -> NDArray[float]:
        return np.array(
            [
                self.feat_linearity,
                self.feat_planarity,
                self.feat_scattering,
                # self.feat_omnivariance,  # FIXME: sqrt of negative number?
                self.feat_anisotropy,
                self.feat_eigentropy,
                self.feat_eigensum,
                self.feat_change_in_curvature,
            ],
        )


def get_features(points3d: NDArray[float], scale: float) -> NDArray[float]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d[:, :3])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    features = []
    # TODO: Use multiprocessing
    for pt3d in tqdm(pcd.points, desc=f'Calculating features for 3D points at scale {scale}:'):
        idx = pcd_tree.search_radius_vector_3d(pt3d, scale)[1]
        features.append(LocalPCD.from_pt_and_neighbours(pcd, pt3d, idx).features)

    return np.vstack(features)
