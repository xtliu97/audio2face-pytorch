import torch
import torch.nn as nn


class AutoCorrelation(nn.Module):
    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        x = torch.matmul(x, x.transpose(1, 2))
        return x


class Audio2Face(nn.Module):
    """https://research.nvidia.com/sites/default/files/publications/karras2017siggraph-paper_0.pdf"""

    def __init__(self, out_nums: int = 2018):
        super(Audio2Face, self).__init__()
        self.analysis_net = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
        )

        self.articulation_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=(4, 1)),
            nn.ReLU(),
        )

        self.output_net = nn.Sequential(
            nn.Linear(256, 150), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(150, out_nums)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.analysis_net(x)
        x = self.articulation_net(x)
        x = x.view(x.size(0), -1)
        x = self.output_net(x)
        return x


if __name__ == "__main__":
    model = Audio2Face()
    x = torch.rand(3, 32, 64).reshape(3, 64, 32)
    print(model(x).shape)
