#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli, filter_frame_corners
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class _CornerTracker:
    PYRAMID_LEVELS = 3
    BLOCK_SIZE = 10
    RADIOUS = 7
    MAX_CORNERS = 1000
    QUALITY_LEVEL = 0.01
    MIN_DISTANCE = 2

    def __init__(self, frame_sequence: pims.FramesSequence):
        self._frame_sequence = frame_sequence
        self._last_id = 0

    def find_corners(self):
        image_0 = self._frame_sequence[0]
        corners = self._process_frame(image_0)
        yield corners
        for image_1 in self._frame_sequence[1:]:
            tracked_corners = self._track_old_corners(image_1, image_0, corners)
            new_corners = self._process_frame(image_1, tracked_corners.points)
            if new_corners is not None:
                corners = FrameCorners(
                    ids=np.concatenate((tracked_corners.ids, new_corners.ids)).astype(np.int64),
                    points=np.concatenate((tracked_corners.points, new_corners.points)),
                    sizes=np.concatenate((tracked_corners.sizes, new_corners.sizes))
                )
            else:
                corners = tracked_corners
            image_0 = image_1
            yield corners

    def _process_frame(self, image, points_to_exclude=None):
        number_of_found_corners = 0 if points_to_exclude is None else points_to_exclude.shape[0]
        shape = image.shape
        points, indices, sizes = [], [], []
        block_size = self.BLOCK_SIZE
        for i in range(self.PYRAMID_LEVELS):
            if number_of_found_corners >= self.MAX_CORNERS:
                break
            mask = self._create_mask(points_to_exclude, shape)[::2 ** i, ::2 ** i]
            found_points = cv2.goodFeaturesToTrack(image=image,
                                                   maxCorners=self.MAX_CORNERS - number_of_found_corners,
                                                   qualityLevel=self.QUALITY_LEVEL,
                                                   minDistance=self.MIN_DISTANCE,
                                                   blockSize=self.BLOCK_SIZE,
                                                   mask=mask)
            found_points *= 2 ** i
            found_points = found_points.reshape(-1, 2)
            number_of_found_corners += found_points.shape[0]
            cur_sizes = np.full(shape=found_points.shape[0], fill_value=block_size)
            cur_indices = np.arange(self._last_id, self._last_id + len(found_points))
            self._last_id += len(found_points)
            points.append(found_points)
            sizes.append(cur_sizes)
            indices.append(cur_indices.astype(np.int64))
            block_size *= 2
            if points_to_exclude is not None:
                points_to_exclude = np.concatenate((points_to_exclude, found_points))
            else:
                points_to_exclude = found_points
            image = cv2.pyrDown(image)
        if len(points) == 0:
            return None
        return FrameCorners(ids=np.concatenate(tuple(indices)).astype(np.int64),
                            points=np.concatenate(tuple(points)),
                            sizes=np.concatenate(tuple(sizes)))

    @staticmethod
    def _create_mask(points_to_exclude, shape):
        res = np.full(shape=shape, fill_value=255, dtype=np.uint8)
        if points_to_exclude is None:
            return res
        for p in points_to_exclude:
            cv2.circle(res, center=tuple(p),
                       radius=_CornerTracker.RADIOUS, color=0)
        return res

    @staticmethod
    def _track_old_corners(image, prev_image, prev_corners):
        def to_uint8(img):
            return np.array(img * 255, dtype=np.uint8)

        image = to_uint8(image)
        prev_image = to_uint8(prev_image)
        coords, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=prev_image,
                                                     nextImg=image,
                                                     prevPts=prev_corners.points,
                                                     nextPts=None)
        status = status.ravel()
        tracked_corners = filter_frame_corners(prev_corners, status == 1)
        return FrameCorners(points=coords[status == 1],
                            ids=tracked_corners.ids.astype(np.int64),
                            sizes=tracked_corners.sizes)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    ct = _CornerTracker(frame_sequence)
    for i, c in enumerate(ct.find_corners()):
        builder.set_corners_at_frame(i, c)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
