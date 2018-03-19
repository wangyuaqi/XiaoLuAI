import pandas as pd
import numpy as np

from mtcnet.cfg import cfg


def split_by_gender():
    df = pd.read_csv(cfg['SCUT_FBP5500_txt'], sep=' ', index_col=False, header=None)
    filenames = df[0].tolist()
    scores = df[1].tolist()

    result = []

    for i, filename in enumerate(filenames):
        if filename.startwith('f'):
            result.append([filename, 'F'])


if __name__ == '__main__':
    split_by_gender()
