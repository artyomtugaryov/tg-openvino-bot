import os

ROOT_PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))

DEFAULT_DATA_PATH = os.path.join(ROOT_PROJECT_PATH, 'data')

DATA_DIRECTORY = os.getenv('DATA_PATH', DEFAULT_DATA_PATH)

MODELS_PATH = os.path.join(DATA_DIRECTORY, 'models')
IMAGES_PATH = os.path.join(DATA_DIRECTORY, 'images')
