import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from skimage import io
from skimage.transform import resize

import torch
import torch.nn as nn

sys.path.append('../')
from hmtnet.models import HMTNet


def viz_race(wj=True, hmt_model='./model/hmt-net.pth'):
    """
    visualize race representation in high dimension
    :param wj: with joint-training or not
    :return:
    """
    hmt_net = HMTNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        hmt_net = nn.DataParallel(hmt_net)

    hmt_net.load_state_dict(torch.load(hmt_model))
    hmt_net.to(device)
    hmt_net.eval()

    X = np.zeros([4, 4])  # just for test
    tsne = TSNE(n_components=2)
    tsne.fit_transform(X)

    img_path = "E:/DataSet/Face/SCUT-FBP5500/Images/ftw8.jpg"
    image = resize(io.imread(img_path), (224, 224), mode='constant')
    image[:, :, 0] -= 131.45376586914062
    image[:, :, 1] -= 103.98748016357422
    image[:, :, 2] -= 91.46234893798828

    image = np.transpose(image, [2, 0, 1])
    input = torch.from_numpy(image).unsqueeze(0).float()
    input = input.to(device)

    for idx, module in hmt_net.named_children():
        if idx != 'conv4':
            input = module(input)
        else:
            for id, mod in module.named_children():
                if id == 'rrelu2':
                    # get the feature representation
                    print(input.shape)
                else:
                    input = mod(input)


if __name__ == '__main__':
    viz_race()
