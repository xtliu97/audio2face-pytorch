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
        super().__init__()
        # self.instance_norm = nn.InstanceNorm2d(1, affine=True)

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
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_nums),
        )

        # self.__init_params()

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.kaiming_normal_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.kaiming_normal_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x = self.instance_norm(x)
        x = self.analysis_net(x)
        x = self.articulation_net(x)
        x = x.view(x.size(0), -1)
        x = self.output_net(x)
        return x


class Audio2FaceLSTM(nn.Module):

    def __init__(self, out_nums=51, num_emotions=16):
        super().__init__()

        self.num_blendshapes = out_nums
        self.num_emotions = num_emotions

        # emotion network with LSTM
        self.emotion = nn.LSTM(
            input_size=32,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.dense = nn.Sequential(
            nn.Linear(128 * 2, 150), nn.ReLU(), nn.Linear(150, self.num_emotions)
        )

        # formant analysis network
        self.formant = nn.Sequential(
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

        # articulation network
        self.conv1 = nn.Conv2d(
            256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )
        self.conv2 = nn.Conv2d(
            256 + self.num_emotions,
            256,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
        )
        self.conv5 = nn.Conv2d(
            256 + self.num_emotions, 256, kernel_size=(4, 1), stride=(4, 1)
        )
        self.relu = nn.ReLU()

        # output network
        self.output = nn.Sequential(
            nn.Linear(256 + self.num_emotions, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_blendshapes),
        )

    def forward(self, x):
        # extract emotion state
        e_state, _ = self.emotion(x[:, ::2])  # input features are 2* overlapping
        e_state = self.dense(e_state[:, -1, :])  # last
        e_state = e_state.view(-1, self.num_emotions, 1, 1)

        x = torch.unsqueeze(x, dim=1)
        # convolution
        x = self.formant(x)

        # conv+concat
        x = self.relu(self.conv1(x))
        x = torch.cat((x, e_state.repeat(1, 1, 32, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 16, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 8, 1)), 1)

        x = self.relu(self.conv2(x))
        x = torch.cat((x, e_state.repeat(1, 1, 4, 1)), 1)

        x = self.relu(self.conv5(x))
        x = torch.cat((x, e_state), 1)

        # fully connected
        x = x.view(-1, 256 + self.num_emotions)
        x = self.output(x)

        return x


if __name__ == "__main__":
    model = Audio2Face()
    x = torch.rand(3, 32, 64).reshape(3, 64, 32)
    print(model(x).shape)
