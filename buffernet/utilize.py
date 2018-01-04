import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


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


def load_config_by_dataset_name(dataset_name='MNIST'):
    """
    load config by dataset name
    :param dataset_name:
    :param config_json_path:
    :return:
    """
    for _ in load_benchmark_config()['dataset']:
        if _['name'] == dataset_name:
            cfg = _
            break

    return cfg


def prepare_data(cfg, transform=transforms.Compose(
    [
        transforms.ColorJitter(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])):
    """
    prepare train and test dataset
    :param cfg:
    :param transform:
    :return:
    """
    if cfg['name'] == "MNIST":
        trainset = torchvision.datasets.MNIST(root=cfg['root'], download=False, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)

        testset = torchvision.datasets.MNIST(root=cfg['root'], download=False, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    elif cfg['name'] == "SVHN":
        trainset = torchvision.datasets.SVHN(root=cfg['root'], split="train", download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)

        testset = torchvision.datasets.SVHN(root=cfg['root'], split="test", download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    else:
        print('Invalid dataset !!')

    return trainloader, testloader


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
