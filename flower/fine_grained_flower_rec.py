import os
import sys
from pprint import pprint

import numpy as np
import scipy.io as sio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from flower import config


def load_data():
    print(config.base_dir)
    filenames = [os.path.join(config.base_dir, 'jpg', _) for _ in
                 np.loadtxt(os.path.join(config.base_dir, 'jpg', 'files.txt'), dtype=str)]
    data = sio.loadmat(os.path.join(config.base_dir, 'datasplits.mat'))
    pprint(data)
    for k, v in data.items():
        if k.startswith('trn'):
            print(k)


if __name__ == '__main__':
    load_data()
