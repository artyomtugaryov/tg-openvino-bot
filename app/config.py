from typing import Type


class BaseConfig:
    CONFIG_NAME = 'base'
    USE_MOCK_EQUIVALENCY = False
    DEBUG = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class DevelopmentConfig(BaseConfig):
    CONFIG_NAME = 'dev'
    DEBUG = True
    TESTING = False


def get_config() -> Type[BaseConfig]:
    return DevelopmentConfig
