#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp

import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose, build_correspondences, triangulate_correspondences, TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4
)
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    random.seed(11071998)
    frame_count = len(corner_storage)
    point_cloud_builder = PointCloudBuilder()
    calc_ids = [known_view_1[0], known_view_2[0]]
    remained_ids = [idx for idx in range(frame_count) if idx not in calc_ids]
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    def _find_cloud_points(reprojection_error=5,
                           angel=1.5,
                           depth=0.0):
        known1 = calc_ids[-1]
        for known2 in calc_ids[:-1]:
            cors = build_correspondences(
                corner_storage[known1],
                corner_storage[known2],
                ids_to_remove=point_cloud_builder.ids.astype(np.int32)
            )
            print(f'Found {len(cors.ids)} correspondences for frames {known1}, {known2}')
            if len(cors.ids) == 0:
                continue
            points3d, ids, med_cos = triangulate_correspondences(correspondences=cors,
                                                                 intrinsic_mat=intrinsic_mat,
                                                                 view_mat_1=view_mats[known1],
                                                                 view_mat_2=view_mats[known2],
                                                                 parameters=TriangulationParameters(
                                                                     max_reprojection_error=reprojection_error,
                                                                     min_triangulation_angle_deg=angel,
                                                                     min_depth=depth
                                                                 ))
            point_cloud_builder.add_points(ids=ids.astype(np.int64), points=points3d)
            print(f'New cloud points from triangulation {len(ids)}, '
                  f'total number of cloud points {len(point_cloud_builder.points)}\n')

    _find_cloud_points(reprojection_error=10, angel=0.5)
    while len(remained_ids) != 0:
        unknown = random.choice(remained_ids)
        print(f'Points to process: {len(remained_ids)}')
        print(f'Processing frame {unknown}')
        _, (corn_ids, pc_ids) = snp.intersect(corner_storage[unknown].ids.flatten().astype(np.int64),
                                              point_cloud_builder.ids.flatten(),
                                              indices=True)
        if len(corn_ids) < 6:
            print("Not enough points for PnP\n")
            continue
        points = point_cloud_builder.points[pc_ids]
        corners = corner_storage[unknown].points[corn_ids]
        retval, _, _, inliers = cv2.solvePnPRansac(points,
                                                   corners,
                                                   cameraMatrix=intrinsic_mat,
                                                   distCoeffs=None)
        if not retval:
            print("PnP RANSAC failed\n")
            continue

        print(f'Number of inliers {len(inliers) if inliers is not None else 0}')
        if inliers is None or len(inliers) < 6:
            print("Not enough points for PnP\n")
            continue

        points = points[inliers]
        corners = corners[inliers]
        retval, rvec, tvec = cv2.solvePnP(points,
                                          corners,
                                          cameraMatrix=intrinsic_mat,
                                          distCoeffs=None,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        if not retval:
            print("PnP failed\n")
            continue

        print(f'Processing frame {unknown} succeed\n')
        view_mats[unknown] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        calc_ids.append(unknown)
        remained_ids.remove(unknown)
        _find_cloud_points()

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
