from itertools import product as product
from typing import List

import cv2
import numpy as np

from services.inference import Data, IDataProcessor
from services.inference.face_detection.data import FaceDetectionInferenceResultData


class ImageResizePreProcessor(IDataProcessor):

    def __init__(self, height: int = 640):
        self._height = height

    def process(self, data: Data) -> Data:
        data = data.data

        dim = (self._height, self._height)

        # resize the image
        resized = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)

        return Data(resized)


class ImageBGRToRGBPreProcessor(IDataProcessor):

    def process(self, data: Data) -> Data:
        data = data.data
        rgb_data = data[:, :, ::-1].copy()
        return Data(rgb_data)


class ImageHWCToCHWPreProcessor(IDataProcessor):

    def process(self, data: Data) -> Data:
        data = data.data
        transposed_data = data.transpose((2, 0, 1))
        return Data(transposed_data)


class ExpandDimImagePreProcessor(IDataProcessor):

    def process(self, data: Data) -> Data:
        data = data.data
        expanded_data = np.expand_dims(data, axis=0)
        return Data(expanded_data)


def nms(x1, y1, x2, y2, scores, thresh: float = 0.3, include_boundaries=True, keep_top_k=None):
    b = 1 if include_boundaries else 0
    areas = (x2 - x1 + b) * (y2 - y1 + b)
    order = scores.argsort()[::-1]

    if keep_top_k:
        order = order[:keep_top_k]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + b)
        h = np.maximum(0.0, yy2 - yy1 + b)
        intersection = w * h

        union = (areas[i] + areas[order[1:]] - intersection)
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

        order = order[np.where(overlap <= thresh)[0] + 1]

    return keep


class FaceDetectionPostProcessor(IDataProcessor):
    def __init__(self, original_input_size: List[int], input_layer_shape: List[int]):
        self._original_input_size = original_input_size
        self._input_layer_shape = input_layer_shape
        self._variance = [0.1, 0.2]

    def process(self, data: FaceDetectionInferenceResultData) -> Data:
        prior_data = self.generate_prior_data()
        proposals = self._get_proposals(data.boxes, prior_data)
        scores = data.scores[:, 1]
        filter_idx = np.where(scores > 0.8)[0]
        proposals = proposals[filter_idx]
        scores = scores[filter_idx]
        if np.size(scores) > 0:
            x_mins, y_mins, x_maxs, y_maxs = proposals.T
            keep = nms(x_mins, y_mins, x_maxs, y_maxs, scores)

            proposals = proposals[keep]
            scores = scores[keep]
        scale_x = self._original_input_size[1] / self._input_layer_shape[2]
        scale_y = self._original_input_size[0] / self._input_layer_shape[3]

        result = []
        if np.size(scores) != 0:
            scores = np.reshape(scores, -1)
            x_mins, y_mins, x_maxs, y_maxs = np.array(proposals).T  # pylint: disable=E0633
            x_mins *= scale_x
            x_maxs *= scale_x
            y_mins *= scale_y
            y_maxs *= scale_y

            for x_min, y_min, x_max, y_max, score in zip(x_mins, y_mins, x_maxs, y_maxs, scores):
                result.append((x_min, y_min, x_max, y_max, score))

        return Data(result)

    def generate_prior_data(self):
        global_min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        anchors = []
        input_shape_h = self._input_layer_shape[2]
        input_shape_w = self._input_layer_shape[3]
        feature_maps = [
            [int(np.rint(input_shape_h / step)), int(np.rint(input_shape_w / step))] for
            step in steps]
        for idx, feature_map in enumerate(feature_maps):
            min_sizes = global_min_sizes[idx]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes:
                    s_ky = min_size / input_shape_h
                    s_kx = min_size / input_shape_w
                    dense_cx = [x * steps[idx] / input_shape_w for x in [j + 0.5]]
                    dense_cy = [y * steps[idx] / input_shape_h for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        priors = np.array(anchors).reshape((-1, 4))
        return priors

    def _get_proposals(self, raw_boxes, priors):
        proposals = self.decode_boxes(raw_boxes, priors)
        proposals[:, ::2] = proposals[:, ::2] * self._input_layer_shape[2]
        proposals[:, 1::2] = proposals[:, 1::2] * self._input_layer_shape[3]
        return proposals

    def decode_boxes(self, raw_boxes, priors):
        boxes = np.concatenate((
            priors[:, :2] + raw_boxes[:, :2] * self._variance[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(raw_boxes[:, 2:] * self._variance[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes


class DrawFaceBoxes(IDataProcessor):
    def __init__(self, image_data: Data):
        super().__init__()
        self._image_data = image_data
        # self._input_layer_shape = input_layer_shape

    def process(self, data: Data) -> Data:
        detected_faces = data.data
        # scale_x = self._image_data.shape[1] / self._input_layer_shape[2]
        # scale_y = self._image_data.shape[0] / self._input_layer_shape[2]
        image = self._image_data.data
        for face in detected_faces:
            score = str(face[4])
            x_min = int(face[0])
            y_min = int(face[1])
            x_max = int(face[2])
            y_max = int(face[3])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, score, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return self._image_data
