import os

import yaml

from common.singleton import Singleton


class Config(object):
    __metaclass__ = Singleton
    CONF = dict()

    def __init__(self, config_file_path=None):
        self.CONF = self.load_config(config_file_path)

    @classmethod
    def load_config(cls, config_file_path=None):
        if not config_file_path:
            config_file_path = os.path.dirname(os.path.abspath(__file__))
            config_file_path = os.path.join(config_file_path, 'config.yml')
        with open(config_file_path, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)

        for category in ["directory", "path"]:
            for key in config[category]:
                relative_path = config[category][key]
                config[category][key] = os.path.abspath(relative_path)

        return config
