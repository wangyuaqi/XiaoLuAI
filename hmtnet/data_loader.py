import sys

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append('../')
from hmtnet.cfg import cfg


def split_train_and_test_with_py_datasets(data_set, batch_size=cfg['batch_size'], test_size=0.2, num_works=4,
                                          pin_memory=True):
    """
    split datasets into train and test loader
    :param data_set:
    :param batch_size:
    :param test_size:
    :param num_works:
    :param pin_memory:
    :return:
    """
    num_dataset = len(data_set)
    indices = list(range(num_dataset))
    split = int(np.floor(test_size * num_dataset))

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_works,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_works,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
