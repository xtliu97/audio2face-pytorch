# %% import
import os

import numpy as np
from rich.progress import track
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tensorboardX as tb

# import pytorch_lightning as pl

from dataset.vocaset import VocaSet
from model.audio2bs import Audio2Face
from utils.renderer import Renderer, FaceMesh, images_to_video
from trainer.trainer import Trainer

# %% load dataset
dataset_path = os.getcwd()
trainset = VocaSet(dataset_path, "train")
valset = VocaSet(dataset_path, "val")
print(f"Trainset size: {len(trainset)}")
print(f"Valset size: {len(valset)}")

sample_verts = FaceMesh.load("assets/FLAME_sample.obj").verts

EXPNAME = "loss_1_100_lre-5"


# %% load model
def collate_fn(batch):
    inputs = []
    labels = []
    for sample in batch:
        # inputs.append(sample.feature)
        inputs.append(sample.feature[:, :64].transpose(0, 1))

        # interpolate to (32, 64)
        # inputs.append(
        #     F.interpolate(
        #         torch.tensor(sample.feature, dtype=torch.float32)
        #         .unsqueeze(0)
        #         .unsqueeze(0),
        #         size=(32, 64),
        #     )
        #     .squeeze(0)
        #     .squeeze(0)
        #     .transpose(0, 1)
        # )
        labels.append(
            torch.tensor(sample.verts).reshape(-1).float() * 100
            - torch.tensor(sample_verts).reshape(-1).float() * 100
        )
    inputs = torch.stack(inputs)
    # normalize
    # transform = transforms.Compose([transforms.Normalize(mean=0, std=1)])
    # inputs = transform(inputs)
    labels = torch.stack(labels)
    return inputs, labels


# %% trainer
model = Audio2Face(5023 * 3)
dataset_path = os.getcwd()
trainset = VocaSet(dataset_path, "train")
valset = VocaSet(dataset_path, "val")
batch_size = 64

train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    valset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True,
)

# %%


def inference():
    texture_mesh = FaceMesh.load("assets/FLAME_sample.obj")
    renderer = Renderer(texture_mesh)

    model = Audio2Face(5023 * 3)
    model.load_state_dict(torch.load(f"ckpt/{EXPNAME}/best_model.pth"))
    model.eval()
    dataset_path = os.getcwd()
    valset = VocaSet(dataset_path, "test")
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
            gts.append(labels.reshape(-1, 5023, 3).numpy() / 100 + sample_verts)
            preds.append(outputs.reshape(-1, 5023, 3).numpy() / 100 + sample_verts)
            if i > 100:
                break
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    def info(tensors):
        print(
            f"shape: {tensors.shape}, max: {np.max(tensors)}, min: {np.min(tensors)}, std: {np.std(tensors)}, mean: {np.mean(tensors)}, sum: {np.sum(tensors)}, abs_sum: {np.sum(np.abs(tensors))}"
        )

    info(gts)
    info(preds)

    # gts = renderer.render(gts)
    preds = renderer.render(preds)
    # images_to_video(gts, "gt.mp4")
    images_to_video(preds, f"{EXPNAME}_pred.mp4")


trainer = Trainer(EXPNAME)
trainer.run(model, train_loader, val_loader)
inference()

# %%
