import torch
import torch.nn as nn
import pytorch_lightning as pl


class AutoCorrelation(nn.Module):
    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        x = torch.matmul(x, x.transpose(1, 2))
        return x


class PaperLoss:
    def __call__(self, pred, gt):
        loss_p = torch.mean((pred - gt) ** 2)
        split_pred = torch.split(pred, 2, dim=0)
        split_gt = torch.split(gt, 2, dim=0)
        loss_m = 2 * torch.mean(
            (split_pred[0] - split_pred[1] - (split_gt[0] - split_gt[1])) ** 2
        )
        return loss_p + loss_m


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


class Audio2FaceModel(pl.LightningModule):
    def __init__(self, out_nums: int = 5023 * 3):
        super(Audio2FaceModel, self).__init__()
        self.model = Audio2Face(out_nums)
        self.criterion = PaperLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        print(f"val_loss: {self.trainer.callback_metrics['val_loss']}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer  

if __name__ == "__main__":
    model = Audio2Face()
    x = torch.rand(3, 32, 64).reshape(3, 64, 32)
    print(model(x).shape)
