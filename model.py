from torch import nn


class SentimentPLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(768, 256), nn.ReLU(),
                                 nn.Linear(256, 16), nn.ReLU(),
                                 nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x
