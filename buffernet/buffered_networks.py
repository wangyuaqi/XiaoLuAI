from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F


class BufferMLP(nn.Module):
    """
    BufferNet with MLP for SVHN
    """

    def __init__(self):
        super(BufferMLP, self).__init__()
        self.bf1 = nn.Linear(32 * 32 * 3, 28 * 28)
        self.sl1 = nn.Linear(28 * 28, 128)
        self.sl2 = nn.Linear(128, 64)
        self.sl3 = nn.Linear(64, 32)
        self.sl4 = nn.Linear(32, 10)
        # self.layers = nn.Sequential(OrderedDict([
        #     # 1st buffered layer
        #     ('bf1', nn.Sequential(nn.Linear(32 * 32 * 3, 28 * 28),
        #                           nn.ReLU())),
        #     # 1st solid layer
        #     ('sl1/fc1', nn.Sequential(nn.Linear(28 * 28, 128),
        #                               nn.ReLU())),
        #     # 2nd solid layer
        #     ('sl2/fc2', nn.Sequential(nn.Linear(128, 64),
        #                               nn.ReLU())),
        #     ('sl3/fc3', nn.Sequential(nn.Linear(64, 32),
        #                               nn.ReLU())),
        #     ('sl4/fc4', nn.Sequential(nn.Linear(32, 10)))
        # ]))

    def forward(self, x):
        num_features = self.num_flat_features(x)
        x = x.view(-1, num_features)
        x = F.relu(self.bf1(x))
        x = F.relu(self.sl1(x))
        x = F.relu(self.sl2(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.sl3(x))
        x = F.dropout(x, 0.5)
        x = self.sl4(x)

        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self, params):
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p

        return params
