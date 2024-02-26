import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import tensorboardX as tb

import dataclasses

from rich import print
from rich.progress import track


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclasses.dataclass
class Hyperparameters:
    learning_rate: float = 0.001
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Loss:
    def __init__(self):
        pass

    def reconstruction_loss(self, pred, gt):
        return torch.mean(torch.sum((pred - gt) ** 2, axis=2))

    def velocity_loss(self, pred, gt):
        n_consecutive_frames = 2
        pred = pred.view(-1, n_consecutive_frames, self.n_verts, 3)
        gt = gt.view(-1, n_consecutive_frames, self.n_verts, 3)

        v_pred = pred[:, 1] - pred[:, 0]
        v_gt = gt[:, 1] - gt[:, 0]

        return torch.mean(torch.sum((v_pred - v_gt) ** 2, axis=2))

    def verts_reg_loss(self, pred, gt):
        raise

    def __call__(self, pred, gt):
        bs = pred.shape[0]
        gt = gt.view(bs, -1, 3)
        pred = pred.view(bs, -1, 3)
        self.n_verts = pred.shape[1]

        return {
            "loss": self.reconstruction_loss(pred, gt),
            "loss_motion": self.velocity_loss(pred, gt) * 10,
        }


class Trainer:
    def __init__(self, exp_name: str = "default"):
        self.exp_name = exp_name
        self.device = torch.device("cuda")
        self.epochs = 50
        self.epoch = 0
        self.learning_rate = 1e-4
        self.criterion = Loss()
        self.log_step_interval = 10
        self.logger = tb.SummaryWriter(f"logs/{self.exp_name}")
        self.model_ckpt_path = f"ckpt/{self.exp_name}"
        os.makedirs(self.model_ckpt_path, exist_ok=True)
        self.best_val_loss = float("inf")
        self.best_val_epoch = 0

    def load_optimizer(self, model: nn.Module, learning_rate: float):
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

    def run(
        self,
        model: nn.Module,
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
    ):
        self.model = model
        self.model.to(self.device)
        # init params

        self.load_optimizer(self.model, self.learning_rate)
        for epoch in range(self.epochs):
            self.epoch = epoch
            print(f"Running epoch {self.epoch} / {self.epochs}")
            self.train(train_loader)
            self.validate(val_loader)

    def train(self, train_loader: data.DataLoader):
        self.model.train()
        for step, batchdata in enumerate(
            track(train_loader, description=f"Training epoch {self.epoch}")
        ):
            inputs, onehots, targets, template_verts = (
                batchdata["inputs"].to(self.device),
                batchdata["onehots"].to(self.device),
                batchdata["labels"].to(self.device),
                batchdata["template_vert"].to(self.device),
            )
            self.optimizer.zero_grad()
            outputs = self.model(inputs, onehots, template_verts)
            raw_loss = self.criterion(outputs, targets)
            loss = raw_loss["loss"] + raw_loss["loss_motion"]
            if step % self.log_step_interval == 0:
                print(
                    f"Epoch {self.epoch} / {self.epochs} Step {step} / {len(train_loader)} "
                    f"Loss: {raw_loss['loss'].item()} "
                    f"Loss Motion: {raw_loss['loss_motion'].item()}"
                )
                self.logger.add_scalar(
                    "train_loss/loss",
                    raw_loss["loss"].item(),
                    self.epoch * len(train_loader) + step,
                )
                self.logger.add_scalar(
                    "train_loss/loss_motion",
                    raw_loss["loss_motion"].item(),
                    self.epoch * len(train_loader) + step,
                )
            # calculate gradients
            loss.backward()
            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(
            #     self.model.parameters(), max_norm=0.01, norm_type=2
            # )
            self.optimizer.step()

        # save model
        torch.save(self.model.state_dict(), f"ckpt/{self.exp_name}/latest.pth")
        # train metrics
        train_loss = self.metric(train_loader)
        print(f"Training loss: {train_loss}")
        self.logger.add_scalar("MSE/train", train_loss, self.epoch)

    def validate(self, val_loader: data.DataLoader):
        val_loss = self.metric(val_loader)
        print(f"Validation loss: {val_loss}")
        self.logger.add_scalar("MSE/val", val_loss, self.epoch)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_epoch = self.epoch
            print("Best model found! Saving model...")
            torch.save(self.model.state_dict(), f"ckpt/{self.exp_name}/best.pth")

        print(f"Best val loss: {self.best_val_loss} at epoch {self.best_val_epoch}")

    @torch.no_grad()
    def metric(self, dataloader: data.DataLoader):
        self.model.eval()
        all_loss = 0
        with torch.no_grad():
            for batchdata in track(
                dataloader, description=f"Validating epoch {self.epoch}"
            ):
                inputs, onehots, targets, template_verts = (
                    batchdata["inputs"].to(self.device),
                    batchdata["onehots"].to(self.device),
                    batchdata["labels"].to(self.device),
                    batchdata["template_vert"].to(self.device),
                )
                outputs = self.model(inputs, onehots, template_verts)
                # raw_loss = self.criterion(outputs, targets)
                loss = torch.nn.functional.mse_loss(outputs, targets)
                all_loss += loss.item()
        return all_loss / len(dataloader)


if __name__ == "__main__":

    gt = torch.stack([i * i * torch.ones(5023, 3) for i in range(10)])
    pred = torch.stack([i * torch.ones(5023, 3) for i in range(10)])
    loss = Loss()
    print(loss(gt, pred))
