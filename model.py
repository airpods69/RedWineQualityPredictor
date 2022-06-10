import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):

    def __init__(self, n_features):
        super(Model, self).__init__()

        self.layer1 = nn.Linear(n_features, n_features)
        self.layer2 = nn.Linear(n_features, 6) # 6 output features

    def forward(self, data):

        activation1 = self.layer1(data)
        activation1 = torch.sigmoid(activation1)

        activation2 = self.layer2(activation1)

        return torch.sigmoid(activation2)
