from pathlib import Path
from typing import Tuple

import tensorflow as tf
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=Path, required=True)
    parser.add_argument('--input-image', '-i', type=Path, required=True)
    parser.add_argument('--output-image', '-o', type=Path, required=False)
    args = parser.parse_args()
    if not args.output_image:
        input_image_directory = Path(args.input_image).parent
        args.output_image = input_image_directory / 'out.bmp'
    return args


def convert2float(image):
    '''
    Transfroms from int image ([0,255]) to float tensor ([-1.,1.])

    :param image: Image to transform
    :return: transformed image
    '''

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0


def prepare_images(image_path: Path, shape: Tuple[int] = (256, 256), channels: int = 3):
    with tf.compat.v1.gfile.GFile(str(image_path), 'rb') as image:
        image_data = image.read()
        processed_image = tf.image.decode_jpeg(image_data, channels=channels)
        processed_image = tf.image.resize(processed_image, size=shape)
        processed_image = convert2float(processed_image)
        processed_image.set_shape([*shape, channels])
        return processed_image


def inference(model_path: Path, input_image_path: Path, output_image_path: Path):
    graph = tf.Graph()

    with graph.as_default():
        image = prepare_images(input_image_path)

        with tf.compat.v1.gfile.GFile(str(model_path), 'rb') as model_file:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(model_file.read())
        [output_image] = tf.import_graph_def(graph_def,
                                             input_map={'input_image': image},
                                             return_elements=['output_image:0'],
                                             name='output')

    with tf.compat.v1.Session(graph=graph):
        generated = output_image.eval()
        with open(output_image_path, 'wb') as f:
            f.write(generated)


if __name__ == '__main__':
    arguments = parse_arguments()
    inference(model_path=arguments.model,
              input_image_path=arguments.input_image,
              output_image_path=arguments.output_image)
