import os
import sys
import logging

import torch
import torch.nn as nn
import torchaudio
import lightning as L

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.renderer import Renderer, FaceMesh, images_to_video  # noqa


class MFCCExtractor(nn.Module):
    """
    Input shape: (batch, time)
    Output shape: (batch, ? , n_mfcc)
    """

    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int,
        out_dim: int,
        win_length: int,
        n_fft: int | None = None,
        normalize: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.out_dim = out_dim
        self.win_length = win_length
        self.hop_length = win_length // 2  # 50% overlap
        self.n_fft = n_fft if n_fft else win_length
        self.T = self._get_extractor()
        self.normalize = normalize
        self.__running_for_first_time = True

    def _get_extractor(self):
        return torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "win_length": self.win_length,
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = x / 32768.0
        x = self.T(x).transpose(1, 2)
        if self.out_dim != x.shape[1]:
            if self.__running_for_first_time:
                logging.warning(
                    f"MFCCExtractor: Got tensor shape {x.shape}, resizing to {self.out_dim} using bilinear interpolation"
                )
                self.__running_for_first_time = False
            x = torch.nn.functional.interpolate(
                x.unsqueeze(1), size=(self.out_dim, self.n_mfcc), mode="bilinear"
            ).squeeze(1)
        return x


class Audio2FaceBase(nn.Module):
    """https://research.nvidia.com/sites/default/files/publications/karras2017siggraph-paper_0.pdf"""

    def __init__(self, n_verts: int, n_onehot: int):
        super().__init__()
        self.n_verts = n_verts
        self.n_onehot = n_onehot

        self.wav2feature = MFCCExtractor(
            sample_rate=22000,
            n_mfcc=32,
            out_dim=52,
            win_length=220 * 2,
            n_fft=1024,
            normalize=True,
        )
        self.analysis_net = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(162),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(243),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.articulation_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=(4, 1)),
            nn.ReLU(),
        )

        self.output_net = nn.Sequential(
            nn.Linear(256 + n_onehot, 72),
            nn.Linear(72, 128),
            nn.Tanh(),
            nn.Linear(128, 50),
            nn.Linear(50, n_verts),
        )

    def forward(self, x, one_hot, template):
        bs = x.size(0)
        x = self.wav2feature(x)
        onehot_embedding = one_hot.repeat(1, 32).view(bs, 1, -1, 32)
        x = x.unsqueeze(1)
        # x = self.instance_norm(x)
        x = self.analysis_net(torch.cat((x, onehot_embedding), 2))
        x = self.articulation_net(x)
        x = x.view(x.size(0), -1)
        x = self.output_net(torch.cat((x, one_hot), 1))
        return x.view(bs, -1, 3) + template


class Loss:
    def __init__(self, k_rec: float = 1.0, k_vel: float = 10.0):
        self.k_rec = k_rec
        self.k_vel = k_vel

    def reconstruction_loss(self, pred, gt):
        return torch.mean(torch.sum((pred - gt) ** 2, axis=2))

    def velocity_loss(self, pred, gt):
        n_consecutive_frames = 2
        pred = pred.view(-1, n_consecutive_frames, self.n_verts, 3)
        gt = gt.view(-1, n_consecutive_frames, self.n_verts, 3)

        v_pred = pred[:, 1] - pred[:, 0]
        v_gt = gt[:, 1] - gt[:, 0]

        return torch.mean(torch.sum((v_pred - v_gt) ** 2, axis=2))

    def __call__(self, pred, gt):
        bs = pred.shape[0]
        gt = gt.view(bs, -1, 3)
        pred = pred.view(bs, -1, 3)
        self.n_verts = pred.shape[1]

        rec_loss = self.reconstruction_loss(pred, gt)
        vel_loss = self.velocity_loss(pred, gt)

        return {
            "loss": rec_loss * self.k_rec + vel_loss * self.k_vel,
            "rec_loss": rec_loss,
            "vel_loss": vel_loss,
        }


class Voca(nn.Module):
    def __init__(self, n_verts: int, n_onehot: int):
        super().__init__()
        self.n_verts = n_verts
        self.n_onehot = n_onehot

        self.feature_extractor = MFCCExtractor(
            sample_rate=22000,
            n_mfcc=16,
            out_dim=29,
            win_length=395 * 2,
            n_fft=2048,
            normalize=True,
        )

        self.time_conv = nn.Sequential(
            nn.Conv2d(37, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64 + 8, 72),
            nn.Linear(72, 128),
            nn.Tanh(),
            nn.Linear(128, 50),
            nn.Linear(50, n_verts),
        )

    def forward(self, x, one_hot, template):
        bs = x.size(0)
        one_hot = one_hot[:, :8]
        x = self.feature_extractor(x)
        onehot_embedding = one_hot.repeat(1, 16).view(bs, 1, -1, 16)
        x = x.unsqueeze(1)
        x = torch.cat((x, onehot_embedding), 2)  # [bs, 1, 37, 16]
        x = x.permute(0, 2, 3, 1)  # [bs, 37, 16,1]
        x = self.time_conv(x)
        x = torch.concat([x.view(bs, -1), one_hot], 1)  # [bs, 64 + 8]
        x = self.decoder(x)
        return x.view(bs, -1, 3) + template


class Audio2Face(L.LightningModule):
    def __init__(self, n_verts: int, n_onehot: int):
        super().__init__()
        self.model = Voca(n_verts, n_onehot)
        self.loss = Loss(k_rec=1.0, k_vel=10)
        self.lr = 1e-4
        self.lr_weight_decay = 1e-5

        self.training_error = []
        self.validation_error = []
        self.validation_pred_verts = []
        self.validation_gt_verts = []

        self.predict_error = []
        self.predicted_verts = []

        self.save_hyperparameters()

    def on_train_epoch_end(self):
        epoch_rec_loss = sum(self.training_error) / len(self.training_error)
        self.log("train/err", epoch_rec_loss, on_epoch=True)
        print(f"Epoch {self.current_epoch} rec_loss: {epoch_rec_loss}")
        self.training_error = []

    def on_validation_epoch_end(self):
        epoch_rec_loss = sum(self.validation_error) / len(self.validation_error)
        self.log("val/err", epoch_rec_loss, on_epoch=True)
        print(f"Epoch {self.current_epoch} val_rec_loss: {epoch_rec_loss}")
        self.validation_error = []

    def unpack_batch(self, batch):
        batch["verts"] = batch["verts"] * 100
        batch["template_vert"] = batch["template_vert"] * 100
        return batch["audio"], batch["one_hot"], batch["verts"], batch["template_vert"]

    def training_step(self, batch, batch_idx):
        x, one_hot, gt, template = self.unpack_batch(batch)
        pred = self.model(x, one_hot, template)
        loss = self.loss(pred, gt)
        self.log_dict(loss, on_epoch=False, on_step=True, prog_bar=True)
        self.training_error.append(loss["rec_loss"].item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, one_hot, gt, template = self.unpack_batch(batch)
        pred = self.model(x, one_hot, template)
        loss = self.loss(pred, gt)
        self.log_dict(loss, on_epoch=False, on_step=True, prog_bar=True)
        self.validation_error.append(loss["rec_loss"].item())
        self.validation_pred_verts.append(pred / 100)
        self.validation_gt_verts.append(gt / 100)

    def on_validation_end(self):
        predicted_verts = torch.cat(self.validation_pred_verts, axis=0)
        gt_verts = torch.cat(self.validation_gt_verts, axis=0)
        self.validation_pred_verts = []
        self.validation_gt_verts = []

        # select one sample to log
        random_idx = torch.randint(0, predicted_verts.shape[0], (1,))

        renderer = self.get_renderer()
        rendered_image = renderer.render(predicted_verts[random_idx].cpu().numpy())[0]
        gt_image = renderer.render(gt_verts[random_idx].cpu().numpy())[0]

        if self.logger:
            self.logger.experiment.add_image(
                "val/prediction", rendered_image.transpose(2, 0, 1), self.current_epoch
            )
            self.logger.experiment.add_image(
                "val/gt", gt_image.transpose(2, 0, 1), self.current_epoch
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.lr_weight_decay
        )
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, one_hot, gt, template = self.unpack_batch(batch)
        pred = self.model(x, one_hot, template)
        self.predict_error.append(
            torch.mean(torch.sum((pred - gt) ** 2, axis=2)).item()
        )
        self.predicted_verts.append(pred / 100)
        return pred

    def get_renderer(self):
        if not hasattr(self, "renderer"):
            self.renderer = Renderer(FaceMesh.load("assets/FLAME_sample.obj"))
        return self.renderer

    def on_predict_epoch_end(self):
        # err
        epoch_rec_loss = sum(self.predict_error) / len(self.predict_error)
        print(f"Epoch {self.current_epoch} predict_rec_loss: {epoch_rec_loss}")
        self.predict_error = []

        # render video
        renderer = self.get_renderer()
        predicted_verts = torch.cat(self.predicted_verts, axis=0)
        self.predicted_verts = []

        rendered_image = renderer.render(predicted_verts.cpu().numpy())

        images_to_video(
            rendered_image, f"{self.trainer.log_dir}/{self.current_epoch}.mp4"
        )


if __name__ == "__main__":
    sample = torch.randn(1, int(0.52 * 22000))
    one_hot = torch.randn(1, 12)
    model = Voca(5023 * 3, 12)
    template = torch.randn(1, 5023, 3)
    out = model(sample, one_hot, template)
    print(out.shape)
