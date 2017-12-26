import os
import json


def mkdirs_if_not_exist(dir_name):
    """
    make directory if not exist
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_config(config_json_path='./bfnet_config.json'):
    """
    load configuration in json file
    :param config_json_path:
    :return:
    """
    with open(config_json_path, mode='rt', encoding='UTF-8') as f:
        config = json.load(f)

    return config
