from typing import Literal

import torch
import lightning as L

from ..loss import VocaLoss
from .voca import Voca
from .audio2face import Audio2Mesh
from ..utils.renderer import Renderer, FaceMesh, images_to_video


class Audio2FaceModel(L.LightningModule):
    def __init__(
        self, model_name: Literal["voca", "audio2mesh"], n_verts: int, n_onehot: int
    ):
        super().__init__()
        if model_name == "voca":
            model = Voca
        elif model_name == "audio2mesh":
            model = Audio2Mesh
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        self.model = model(n_verts, n_onehot)
        self.loss = VocaLoss(k_rec=1.0, k_vel=10)
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

        images_to_video(rendered_image, f"{self.logger.log_dir}/best_ckpt.mp4")
