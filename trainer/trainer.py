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


def diff(verts):
    with torch.no_grad():
        err = 0
        for i in range(1, len(verts)):
            err += torch.sum(torch.abs(verts[i] - verts[i - 1]))
        return err


class Loss:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss()

    def loss_position(self, pred, gt):
        return torch.mean((pred - gt) ** 4)

    def loss_motion(self, pred, gt):
        pred0, pred1 = torch.split(pred, pred.size(0) // 2, dim=0)
        gt0, gt1 = torch.split(gt, gt.size(0) // 2, dim=0)
        return torch.mean((pred1 - pred0 - (gt1 - gt0)) ** 2)

    def __call__(self, pred, gt):
        return {
            "loss": self.loss_position(pred, gt),
            "loss_motion": self.loss_motion(pred, gt) * 100,
        }


class Trainer:
    def __init__(self, exp_name: str = "default"):
        self.exp_name = exp_name
        self.device = torch.device("mps")
        self.epochs = 5
        self.epoch = 0
        self.learning_rate = 0.00001
        self.criterion = Loss()
        self.log_step_interval = 10
        self.logger = tb.SummaryWriter(f"logs/{self.exp_name}")
        self.model_ckpt_path = f"ckpt/{self.exp_name}"
        os.makedirs(self.model_ckpt_path, exist_ok=True)
        self.best_val_loss = float("inf")

    def load_optimizer(self, model: nn.Module, learning_rate: float):
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate  # , weight_decay=1e-5
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
        for step, (inputs, targets) in enumerate(
            track(train_loader, description=f"Training epoch {self.epoch}")
        ):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
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
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1, norm_type=2
            )
            self.optimizer.step()

    def validate(self, val_loader: data.DataLoader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in track(
                val_loader, description=f"Validating epoch {self.epoch}"
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                raw_loss = self.criterion(outputs, targets)
                loss = raw_loss["loss"]
                val_loss += loss.item()
        print(
            f"Epoch {self.epoch} / {self.epochs} "
            f"Val Loss: {val_loss / len(val_loader)} "
        )
        self.logger.add_scalar("val_loss", val_loss / len(val_loader), self.epoch)
        # save model
        torch.save(
            self.model.state_dict(), f"{self.model_ckpt_path}/model_{self.epoch}.pth"
        )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"Best model saved with val loss {val_loss}")
            torch.save(
                self.model.state_dict(), f"{self.model_ckpt_path}/best_model.pth"
            )
