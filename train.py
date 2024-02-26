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

from dataset.vocaset import VocaSet, FrameData
from model.audio2bs import Audio2Face
from utils.renderer import Renderer, FaceMesh, images_to_video
from trainer.trainer import Trainer

# %% load dataset
dataset_path = os.getcwd()
trainset = VocaSet(dataset_path, "train")
valset = VocaSet(dataset_path, "val")
print(f"Trainset size: {len(trainset)}")
print(f"Valset size: {len(valset)}")

EXPNAME = "loss_1p_10m_1e-4_mfcc_onehot_epoch50"


# %% load model
def collate_fn(batch):
    inputs = []
    onehots = []
    labels = []
    templates = []
    for sample in batch:
        sample: FrameData = sample
        # inputs.append(sample.feature)
        # inputs.append(sample.feature[:, :52].transpose(0, 1))

        # interpolate to (32, 64)
        inputs.append(
            F.interpolate(
                torch.FloatTensor(sample.feature).unsqueeze(0).unsqueeze(0),
                # sample.feature.unsqueeze(0).unsqueeze(0),
                size=(32, 52),
            )
            .squeeze(0)
            .squeeze(0)
            .transpose(0, 1)
        )
        labels.append(torch.tensor(sample.verts).reshape(-1).float() * 100)
        onehots.append(torch.tensor(sample.one_hot).float())
        templates.append(torch.tensor(sample.template_vert).float() * 100)
    inputs = torch.stack(inputs)
    onehots = torch.stack(onehots)
    templates = torch.stack(templates)
    # normalize
    # transform = transforms.Compose([transforms.Normalize(mean=0, std=1)])
    # inputs = transform(inputs)
    labels = torch.stack(labels)
    return {
        "inputs": inputs,
        "onehots": onehots,
        "labels": labels,
        "template_vert": templates,
    }


# %% trainer
model = Audio2Face(5023 * 3)
dataset_path = os.getcwd()
trainset = VocaSet(dataset_path, "train")
valset = VocaSet(dataset_path, "val")
batch_size = 256

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
    model.load_state_dict(torch.load(f"ckpt/{EXPNAME}/best.pth"))

    model.eval()
    dataset_path = os.getcwd()
    valset = VocaSet(dataset_path).get_framedatas(
        "FaceTalk_170908_03277_TA", "sentence02"
    )
    val_loader = DataLoader(
        valset, batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True
    )
    device = torch.device("cuda")
    model.to(device)
    gts = []
    preds = []
    with torch.no_grad():
        for i, data in track(enumerate(val_loader), total=len(val_loader)):
            inputs, onehots, labels, template_vert = (
                data["inputs"],
                data["onehots"],
                data["labels"],
                data["template_vert"],
            )
            inputs, onehots, labels, template_vert = (
                inputs.to(device),
                onehots.to(device),
                labels.to(device),
                template_vert.to(device),
            )
            outputs = model(inputs, onehots, template_vert)
            gts.append(labels.reshape(-1, 5023, 3).cpu().numpy() / 100)
            preds.append(outputs.reshape(-1, 5023, 3).cpu().numpy() / 100)
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    def info(tensors):
        print(
            f"shape: {tensors.shape}, max: {np.max(tensors)}, min: {np.min(tensors)}, std: {np.std(tensors)}, mean: {np.mean(tensors)}, sum: {np.sum(tensors)}, abs_sum: {np.sum(np.abs(tensors))}"
        )

    info(gts)
    info(preds)

    gts = renderer.render(gts)
    preds = renderer.render(preds)
    images_to_video(gts, f"{EXPNAME}_gt.mp4")
    images_to_video(preds, f"{EXPNAME}_pred.mp4")


trainer = Trainer(EXPNAME)
trainer.run(model, train_loader, val_loader)
inference()

# %%
