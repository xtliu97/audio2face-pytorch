# %% import
import os

import numpy as np
from rich.progress import track
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorboardX as tb

# import pytorch_lightning as pl

from dataset.vocaset import VocaSet
from model.audio2bs import Audio2Face
from utils.renderer import Renderer, FaceMesh, images_to_video

# %% load dataset
dataset_path = os.getcwd()
trainset = VocaSet(dataset_path, "train")
valset = VocaSet(dataset_path, "val")
print(f"Trainset size: {len(trainset)}")
print(f"Valset size: {len(valset)}")
verts = np.load("assets/verts_sample.npy")
sample_verts = np.mean(verts, axis=0)
print(sample_verts.shape)


# %% load model
def collate_fn(batch):
    inputs = []
    labels = []
    for sample in batch:
        inputs.append(sample.feature[:, :64].transpose(0, 1))
        labels.append(
            torch.tensor(sample.verts).reshape(-1).float()
            - torch.tensor(sample_verts).reshape(-1).float()
        )
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    return inputs, labels


class Loss:
    def __call__(self, pred, gt):
        loss_p = torch.mean((pred - gt) ** 2)
        split_pred = torch.split(pred, 2, dim=0)
        split_gt = torch.split(gt, 2, dim=0)
        loss_m = 2 * torch.mean(
            (split_pred[0] - split_pred[1] - (split_gt[0] - split_gt[1])) ** 2
        )
        return loss_p + loss_m


# %% trainer
model = Audio2Face(5023 * 3)


class Trainer:
    def __init__(self, batch_size=24, lr=1e-3):
        self.model = model
        self.dataset_path = os.getcwd()
        self.trainset = VocaSet(self.dataset_path, "train")
        self.valset = VocaSet(self.dataset_path, "val")
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("mps")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = Loss()
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        self.logger = tb.SummaryWriter("logs")

    def train(self, epochs=1):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    self.logger.add_scalar(
                        "train_loss",
                        running_loss / 100,
                        epoch * len(self.train_loader) + i,
                    )
                    print(
                        f"Epoch {epoch}/{epochs} iter {i}/{len(self.train_loader)} loss: {running_loss / 100}"
                    )
                    running_loss = 0.0
            self.validate(epoch)
            # save model
            torch.save(self.model.state_dict(), f"model_{epoch}.pth")
        print("Finished Training")

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
            self.logger.add_scalar(
                "val_loss", running_loss / len(self.val_loader), epoch
            )
            print(f"Epoch {epoch} val_loss: {running_loss / len(self.val_loader)}")


trainer = Trainer()
trainer.train()
# %%


def inference():
    texture_mesh = FaceMesh.load("assets/FLAME_sample.obj")
    renderer = Renderer(texture_mesh)

    model = Audio2Face(5023 * 3)
    model.load_state_dict(torch.load("model_0.pth"))
    model.eval()
    dataset_path = os.getcwd()
    valset = VocaSet(dataset_path, "val")
    val_loader = DataLoader(
        valset, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True
    )
    device = torch.device("cpu")
    model.to(device)
    gts = []
    preds = []
    with torch.no_grad():
        for i, data in track(enumerate(val_loader), total=len(val_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            gts.append(labels.reshape(-1, 5023, 3).numpy() + sample_verts)
            preds.append(outputs.reshape(-1, 5023, 3).numpy() + sample_verts)
            if i > 200:
                break
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    gts = renderer.render(gts)
    preds = renderer.render(preds)
    images_to_video(gts, "gt.mp4")
    images_to_video(preds, "pred.mp4")


inference()
