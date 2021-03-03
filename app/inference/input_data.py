from typing import TypedDict, Tuple, Type, List

import cv2
import numpy as np

from app.inference.schema.input_info import InputType, SingleInputInfo, InputInfo
from app.inference.service import InferenceService


class RequestLabelData(TypedDict):
    toHairColor: str
    toGender: str
    toYang: bool


class LabelsData:
    hair_colors = {
        'black': 0,
        'blond': 1,
        'brow': 2
    }
    genders = {
        'female': 0,
        'male': 1,
    }

    def __init__(self, request):
        self._data = self._from_request(RequestLabelData(request.form))

    @staticmethod
    def _from_request(request_data: RequestLabelData) -> np.ndarray:
        request_data = RequestLabelData(request_data)
        labels = np.array([0] * 5)
        har_color_index = LabelsData.hair_colors[request_data['toHairColor']]
        labels[har_color_index] = 1
        labels[3] = LabelsData.genders[request_data['toGender']]
        labels[4] = int(request_data['toYang'])
        return labels

    def prepare(self, unused_shape: Tuple[int, int, int, int]) -> np.ndarray:
        return self._data


class ImageData:
    def __init__(self, request):
        self._data = self._from_request(request.files['image'])

    @staticmethod
    def _from_request(request_data) -> np.ndarray:
        data_as_np_array = np.frombuffer(request_data.read(), np.uint8)
        return cv2.imdecode(data_as_np_array, cv2.IMREAD_COLOR)

    def prepare(self, shape: Tuple[int, int, int, int],
                mean: Tuple[int, int, int] = (0.5, 0.5, 0.5),
                std: Tuple[int, int, int] = (0.5, 0.5, 0.5)):
        resized_image = cv2.resize(self._data, shape[-2:])
        rgb_image = resized_image[:, :, ::-1]  #
        chw_rgb_image = rgb_image.transpose((2, 0, 1))  #
        normalized_image = chw_rgb_image / chw_rgb_image.max()
        for channel in range(normalized_image.shape[0]):
            normalized_image[channel] = (normalized_image[channel] - mean[channel]) / std[channel]

        input_data = np.ndarray(shape=shape)
        input_data[0] = normalized_image
        return input_data


class InputData:
    def using(self, inference_service_class: Type[InferenceService]) -> InferenceService:
        inference_service = inference_service_class()
        input_info = inference_service.input_info
        inference_service.input_data = self.prepare_data_for_inference(input_info)
        return inference_service

    def prepare_data_for_inference(self, input_info: InputInfo) -> dict:
        raise NotImplementedError

    def pre_processed(self, shape: Tuple[int, ...]):
        raise NotImplementedError()


class GenderTransformerInputData(InputData):
    def __init__(self, request):
        self._labels = LabelsData(request)
        self._image = ImageData(request)

    def prepare_data_for_inference(self, input_infos: InputInfo) -> dict:
        image_input = input_infos.get_input(InputType.image)
        labels_input_name = input_infos.get_input(InputType.labels).name

        pre_processed_data = self.pre_processed(image_input.shape)

        image_input_data = pre_processed_data.get_input(InputType.image).data
        labels_input_data = pre_processed_data.get_input(InputType.labels).data

        return {
            image_input.name: image_input_data,
            labels_input_name: labels_input_data
        }

    def pre_processed(self, shape: Tuple[int, ...]) -> InputInfo:
        inputs = InputInfo()
        inputs.add_input(
            input_type=InputType.image,
            single_input_info=SingleInputInfo(
                data=self._image.prepare(shape)
            ))
        inputs.add_input(
            input_type=InputType.labels,
            single_input_info=SingleInputInfo(
                data=self._labels.prepare(shape)
            ))
        return inputs
