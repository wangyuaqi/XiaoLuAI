import torch.nn as nn
import torch.nn.functional as F
import torch


class BufferMLP(nn.Module):
    """
    BufferNet with MLP
    """

    def __init__(self):
        super(BufferMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        num_features = self.num_flat_features(x)
        x = F.relu(self.fc1(x.view(-1, num_features)))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))

        return x

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

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['solid_layers']
        self.layers.load_state_dict(shared_layers)
