from typing import Type

import cv2
import numpy as np

from services.inference import Data, IDataProcessor, ImageData


class ImageResizePreProcessor(IDataProcessor):

    def __init__(self, height: int = 300):
        self._height = height

    def process(self, data: Data) -> ImageData:
        data = data.data

        # initialize the dimensions of the image to be resized and
        # grab the image size
        h, w = data.shape[:2]

        # check to see if the width is None

        r = self._height / float(h)
        dim = (int(w * r), self._height)

        # resize the image
        resized = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)

        # return the resized image
        result = ImageData(resized)
        return result


class ImageBGRToRGBPreProcessor(IDataProcessor):

    def process(self, data: Data) -> ImageData:
        data = data.data
        rgb_data = data[:, :, ::-1].copy()
        result = ImageData(rgb_data)
        return result


class ImageRGBToBGRPreProcessor(ImageBGRToRGBPreProcessor):
    pass


class ImageHWCToCHWPreProcessor(IDataProcessor):

    def process(self, data: Data) -> ImageData:
        data = data.data
        transposed_data = data.transpose((2, 0, 1))
        result = ImageData(transposed_data)
        return result


class ImageNormalizePreProcessor(IDataProcessor):

    def __init__(self,
                 mean: np.ndarray = np.array([0.5, 0.5, 0.5]),
                 std: np.ndarray = np.array([0.5, 0.5, 0.5])):
        super().__init__()
        self._mean = mean
        self._std = std

    def process(self, data: Data) -> ImageData:
        data = data.data

        normalized_image = data / data.max()
        for channel in range(normalized_image.shape[0]):
            normalized_image[channel] = (normalized_image[channel] - self._mean[channel]) / self._std[channel]
        result = ImageData(normalized_image)
        return result


class ExpandShapePreProcessor(IDataProcessor):
    def __init__(self, to_class: Type[Data] = Data):
        super().__init__()
        self._to_class = to_class

    def process(self, data: Data) -> Data:
        data = data.data
        shape = data.shape
        expanded_data = np.ndarray(shape=(1, *shape))
        expanded_data[0] = data
        return self._to_class(expanded_data)


class GenderPostProcessor(IDataProcessor):

    def process(self, data: Data) -> Data:
        data = data.data
        key = next(iter(data.keys()))
        data = data[key]
        result = ((np.moveaxis(data[0], [0], [2]) + 1) / 2)
        result *= 255
        return Data(result)
