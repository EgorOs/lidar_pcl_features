import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import open3d as o3d
from numpy.typing import NDArray


@dataclass
class LocalPCD:
    pcd: o3d.geometry.PointCloud = field(default_factory=o3d.geometry.PointCloud)

    @classmethod
    def from_pt_and_neighbours(cls, pcd: o3d.geometry.PointCloud, pt: NDArray[float], nbr_idxs) -> 'LocalPCD':
        nbrs = np.asarray(pcd.points)[nbr_idxs[1:], :]
        local_pcd = o3d.geometry.PointCloud()
        points = np.vstack([pt, nbrs])
        local_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        return cls(local_pcd)

    @cached_property
    def cov_eig(self):
        cov = self.pcd.compute_mean_and_covariance()[1]
        return np.linalg.eig(cov)

    @cached_property
    def feat_linearity(self) -> float:
        e1, e2, e3 = self.cov_eig[0]
        return (e1 - e2) / e1

    @cached_property
    def feat_planarity(self) -> float:
        e1, e2, e3 = self.cov_eig[0]
        return (e2 - e3) / e1

    @cached_property
    def feat_scattering(self) -> float:
        e1, e2, e3 = self.cov_eig[0]
        return e3 / e1

    @cached_property
    def feat_omnivariance(self) -> float:
        e1, e2, e3 = self.cov_eig[0]
        return math.pow((e1 * e2 * e3), 1 / 3)

    @cached_property
    def feat_anisotropy(self) -> float:
        e1, e2, e3 = self.cov_eig[0]
        return (e1 - e3) / e1

    @cached_property
    def feat_eigentropy(self) -> float:
        eigen_vals = self.cov_eig[0]
        return -np.sum(eigen_vals * np.log(eigen_vals))

    @cached_property
    def feat_eigensum(self) -> float:
        return np.sum(self.cov_eig[0])

    @cached_property
    def feat_change_in_curvature(self) -> float:
        e1, e2, e3 = self.cov_eig[0]
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
