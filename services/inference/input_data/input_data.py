from typing import Tuple, Type

import numpy as np

from services.inference.input_data.image import Image
from services.inference.input_data.labels import Labels
from services.inference.schema.input_info import InputType, InputInfo
from services.inference.service import InferenceService


class InputData:
    def __init__(self, image: Image, labels: Labels):
        self._labels = labels
        self._image = image

    def using(self, inference_service_class: Type[InferenceService]) -> InferenceService:
        inference_service = inference_service_class()
        input_info = inference_service.input_info
        self.set_data_to_input_info(input_info)
        inference_service.input_data = input_info
        return inference_service

    def set_data_to_input_info(self, input_info: InputInfo):
        image_input = input_info.get_input(InputType.image)
        labels_input_name = input_info.get_input(InputType.labels)

        image, labels = self.pre_processed()

        image_input.data = image
        labels_input_name.data = labels

    def pre_processed(self, ) -> Tuple[np.ndarray, np.ndarray]:
        prepared_image = self._image.prepare()

        prepared_labels = self._labels.prepare()
        return prepared_image, prepared_labels
