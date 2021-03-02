import numpy as np
import cv2
from openvino.inference_engine import IECore


class SingletonMetaClass(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaClass, cls).__call__(*args, *kwargs)
        return cls._instances[cls]


class InferenceService(metaclass=SingletonMetaClass):
    def __init__(self):
        self.ie_core = IECore()

        network = self.ie_core.read_network(model=str('data/models/horse2zebra.xml'))

        self.input_blob = next(iter(network.input_info))
        self.input_shape = network.input_info[self.input_blob].input_data.shape
        self.output_blob = next(iter(network.outputs))

        self.execution_network = self.ie_core.load_network(network=network, device_name='CPU')

    def do(self):
        image = cv2.imread('data/images/IMG_0945.JPG')
        input_data = self.prepare_image(image, self.input_shape)
        results = self.execution_network.infer(inputs={self.input_blob: input_data})
        result = results[self.output_blob]
        cv2.imwrite('data/images/IMG_0945.bmp', result)
        return []

    @staticmethod
    def prepare_image(image_data, shape) -> np.ndarray:
        image = cv2.resize(image_data, shape[-2:])
        image = image.transpose((2, 0, 1))

        input_data = np.ndarray(shape=shape)
        input_data[0] = image

        return input_data
