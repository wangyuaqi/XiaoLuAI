import os
import json

import torch
import torch.nn as nn
import torch.optim as optim


def mkdirs_if_not_exist(dir_name):
    """
    make directory if not exist
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_benchmark_config(config_json_path='./benchmark_config.json'):
    """
    load benchmark configuration in json file
    :param config_json_path:
    :return:
    """
    with open(config_json_path, mode='rt', encoding='UTF-8') as f:
        config = json.load(f)

    return config


def load_bf_config(config_json_path='./bfnet_config.json'):
    """
    load BufferNet configuration in json file
    :param config_json_path:
    :return:
    """
    with open(config_json_path, mode='rt', encoding='UTF-8') as f:
        config = json.load(f)

    return config


def set_optimizer(model, lr_init, lr_mult, momentum, wd):
    """
    set NN optimizer
    :param model:
    :param lr_init:
    :param lr_mult:
    :param momentum:
    :param wd:
    :return:
    """
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_init
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_init * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=wd)

    return optimizer


def init_weights(m):
    """
    initialize params in Xavier way
    :param model:
    :return:
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)
