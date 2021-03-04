import cv2
import numpy as np

from services.inference.data import Data
from services.inference.data_processor import IDataProcessor
from services.inference.gender.data import InputImageData


class ImageResizePreProcessor(IDataProcessor):

    def process(self) -> InputImageData:
        data = self._data.data
        height = 300
        # initialize the dimensions of the image to be resized and
        # grab the image size
        h, w = data.shape[:2]

        # check to see if the width is None

        r = height / float(h)
        dim = (int(w * r), height)

        # resize the image
        resized = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)

        # return the resized image
        return InputImageData(resized)


class ImageBGRToRGBPreProcessor(IDataProcessor):

    def process(self) -> InputImageData:
        data = self._data.data
        rgb_data = data[:, :, ::-1]
        return InputImageData(rgb_data)


class ImageHWCToCHWPreProcessor(IDataProcessor):

    def process(self) -> InputImageData:
        data = self._data.data
        transposed_data = data.transpose((2, 0, 1))
        return InputImageData(transposed_data)


class ImageNormalizePreProcessor(IDataProcessor):

    def __init__(self, data: InputImageData,
                 mean: np.ndarray = np.array([0.5, 0.5, 0.5]),
                 std: np.ndarray = np.array([0.5, 0.5, 0.5])):
        super().__init__(data=data)
        self._mean = mean
        self._std = std

    def process(self) -> InputImageData:
        data = self._data.data

        normalized_image = data / data.max()
        for channel in range(normalized_image.shape[0]):
            normalized_image[channel] = (normalized_image[channel] - self.mean[channel]) / self.std[channel]
        return InputImageData(normalized_image)


class ExpandShapePreProcessor(IDataProcessor):
    def process(self) -> Data:
        data = self._data.data
        shape = data.shape

        expanded_data = np.ndarray(shape=(1, *shape))
        expanded_data[0] = data
        return Data(expanded_data)


class GenderPostProcessor(IDataProcessor):

    def process(self) -> Data:
        data = self._data.data
        key = next(iter(data.keys()))
        data = data[key]
        result = ((np.moveaxis(data[0], [0], [2]) + 1) / 2)[:, ::-1, ::-1]
        result *= 255
        return Data(result)
