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
    rodrigues_and_translation_to_view_mat3x4, eye3x4, remove_correspondences_with_ids
)
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose


class _CameraTracker:
    DEFAULT_MIN_ANGEL = 1.5
    INITIAL_MIN_ANGEL = 0.5
    HOMOGRAPHY_THRESHOLD = 0.5
    RANSAC_THRESHOLD = 0.5
    DEFAULT_MAX_REPROJECTION_ERROR = 5
    INITIAL_MAX_REPROJECTION_ERROR = 10

    def __init__(self, intrinsic_mat, corner_storage,
                 known_view1, known_view2):
        self._intrinsic_mat = intrinsic_mat
        self._corner_storage = corner_storage
        self._frame_count = len(corner_storage)
        self._view_mats = [eye3x4()] * self._frame_count
        if known_view1 is None or known_view2 is None:
            known_view1, known_view2 = self._initialize_view_mats()
            self._calc_ids = [known_view1, known_view2]
        else:
            self._view_mats[known_view1[0]] = pose_to_view_mat3x4(known_view1[1])
            self._view_mats[known_view2[0]] = pose_to_view_mat3x4(known_view2[1])
            self._calc_ids = [known_view1[0], known_view2[0]]
        self._point_cloud_builder = PointCloudBuilder()
        self._remained_ids = [idx for idx in range(self._frame_count) if idx not in self._calc_ids]
        self._find_cloud_points(reprojection_error=self.INITIAL_MAX_REPROJECTION_ERROR, angel=self.INITIAL_MIN_ANGEL)

    def _find_cloud_points(self, reprojection_error=None,
                           angel=None,
                           depth=0.0):
        if reprojection_error is None:
            reprojection_error = self.DEFAULT_MAX_REPROJECTION_ERROR
        if angel is None:
            angel = self.DEFAULT_MIN_ANGEL
        known1 = self._calc_ids[-1]
        for known2 in self._calc_ids[:-1]:
            cors = build_correspondences(
                self._corner_storage[known1],
                self._corner_storage[known2],
                ids_to_remove=self._point_cloud_builder.ids
            )
            print(f'Found {len(cors.ids)} correspondences for frames {known1}, {known2}')
            if len(cors.ids) == 0:
                continue
            points3d, ids, med_cos = triangulate_correspondences(correspondences=cors,
                                                                 intrinsic_mat=self._intrinsic_mat,
                                                                 view_mat_1=self._view_mats[known1],
                                                                 view_mat_2=self._view_mats[known2],
                                                                 parameters=TriangulationParameters(
                                                                     max_reprojection_error=reprojection_error,
                                                                     min_triangulation_angle_deg=angel,
                                                                     min_depth=depth
                                                                 ))
            self._point_cloud_builder.add_points(ids=ids.astype(np.int64), points=points3d)
            print(f'New cloud points from triangulation {len(ids)}, '
                  f'total number of cloud points {len(self._point_cloud_builder.points)}\n')

    def track(self):
        while len(self._remained_ids) != 0:
            unknown = random.choice(self._remained_ids)
            print(f'Points to process: {len(self._remained_ids)}')
            print(f'Processing frame {unknown}')
            _, (corn_ids, pc_ids) = snp.intersect(self._corner_storage[unknown].ids.flatten().astype(np.int64),
                                                  self._point_cloud_builder.ids.flatten(),
                                                  indices=True)
            if len(corn_ids) < 6:
                print("Not enough points for PnP\n")
                continue
            points = self._point_cloud_builder.points[pc_ids]
            corners = self._corner_storage[unknown].points[corn_ids]
            retval, _, _, inliers = cv2.solvePnPRansac(points,
                                                       corners,
                                                       cameraMatrix=self._intrinsic_mat,
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
                                              cameraMatrix=self._intrinsic_mat,
                                              distCoeffs=None,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
            if not retval:
                print("PnP failed\n")
                continue

            print(f'Processing frame {unknown} succeed\n')
            self._view_mats[unknown] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
            self._calc_ids.append(unknown)
            self._remained_ids.remove(unknown)
            self._find_cloud_points()
        return self._view_mats, self._point_cloud_builder

    def _initialize_view_mats(self):
        print('Calculating initial camera position for two frames')
        frames = list(range(self._frame_count))
        random.shuffle(frames)
        success = 0
        max_points_count = 0
        init_frame1, init_frame2, init_mat = None, None, None
        for frame1_id in range(self._frame_count):
            for frame2_id in range(frame1_id + 1, self._frame_count):
                frame1 = frames[frame1_id]
                frame2 = frames[frame2_id]
                point_count, mat = self._find_camera_position_for_frames(frame1, frame2)
                if mat is not None and point_count > 0:
                    success += 1
                    if max_points_count < point_count:
                        max_points_count = point_count
                        init_mat = mat
                        init_frame1 = frame1
                        init_frame2 = frame2
                    if success > self._frame_count * 10:
                        self._view_mats[init_frame1] = eye3x4()
                        self._view_mats[init_frame2] = init_mat
                        return init_frame1, init_frame2
        self._view_mats[init_frame1] = eye3x4()
        self._view_mats[init_frame2] = init_mat
        return init_frame1, init_frame2

    def _find_camera_position_for_frames(self, frame1, frame2):
        print(f'Check camera position for {frame1}, {frame2}')
        cors = build_correspondences(self._corner_storage[frame1], self._corner_storage[frame2])
        if len(cors.ids) < 5:
            print(f'Not enough correspondences for frames {frame1} {frame2}\n')
            return None, None
        E, mask = cv2.findEssentialMat(cors.points_1,
                                       cors.points_2,
                                       cameraMatrix=self._intrinsic_mat,
                                       method=cv2.RANSAC,
                                       threshold=self.RANSAC_THRESHOLD)
        if E is None or mask is None:
            print('Calculating essential matrix failed')
            return None, None
        cors = remove_correspondences_with_ids(cors, cors.ids[mask.flatten() == 0])
        _, hom_mask = cv2.findHomography(cors.points_1, cors.points_2, method=cv2.RANSAC,
                                         ransacReprojThreshold=self.RANSAC_THRESHOLD, maxIters=1000)
        threshold = np.count_nonzero(hom_mask) / np.count_nonzero(mask) if hom_mask is not None else 1
        if threshold > self.HOMOGRAPHY_THRESHOLD:
            print(f'Homography validation failed, threshold is {threshold}\n')
            return None, None
        retval, R, t, mask = cv2.recoverPose(E, cors.points_1, cors.points_2, self._intrinsic_mat)
        if not retval:
            print(f'Recover pose failed\n')
            return None, None
        mat = np.hstack((R, t))
        points, _, _ = triangulate_correspondences(cors, eye3x4(), mat,
                                                   intrinsic_mat=self._intrinsic_mat,
                                                   parameters=TriangulationParameters(
                                                       self.DEFAULT_MAX_REPROJECTION_ERROR, self.DEFAULT_MIN_ANGEL,
                                                       0.01))
        print(f'{len(points)} points in triangulation\n')
        return len(points), mat


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    random.seed(11071998)
    camera_tracker = _CameraTracker(intrinsic_mat, corner_storage,
                                    known_view_1, known_view_2)
    view_mats, point_cloud_builder = camera_tracker.track()
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
