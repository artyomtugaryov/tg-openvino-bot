from enum import Enum


class InputLayout(Enum):
    nchw = 'NCHW'
    nc = 'NC'


class InputType(Enum):
    image = 'image'
    tensor = 'tensor'
    undefined = 'undefined'


def define_input_type(input_info_data) -> InputType:
    layout_type_map = {
        InputLayout.nc: InputType.tensor,
        InputLayout.nchw: InputType.image,
    }
    try:
        return layout_type_map[InputLayout(input_info_data.layout)]
    except ValueError:
        return InputType.undefined
