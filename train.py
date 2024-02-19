# %% import
import os

import numpy as np
from rich.progress import track
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import tensorboardX as tb

# import pytorch_lightning as pl

from dataset.vocaset import VocaDataModule, VocaSet, collate_fn
from model.audio2bs import Audio2FaceModel
from utils.renderer import Renderer, FaceMesh, images_to_video

# %% load dataset
dataset_path = os.getcwd()
dataset = VocaDataModule(dataset_path, 64)

# %% trainer
model = Audio2FaceModel(5023 * 3)
trainer = pl.Trainer(
    max_epochs=10,
    default_root_dir="logs",
    accelerator="mps",
    callbacks=[ModelCheckpoint(monitor="val_loss", dirpath="ckpts", save_last=True)],
)

# %%
def inference():
    texture_mesh = FaceMesh.load("assets/FLAME_sample.obj")
    renderer = Renderer(texture_mesh)
    device = torch.device("mps")

    model = Audio2FaceModel(5023 * 3)
    model.load_state_dict(torch.load("ckpts/last.ckpt")["state_dict"])
    model.to(device)
    dataset = VocaSet(os.getcwd())
    
    inference_data = dataset.get_framedatas("FaceTalk_170904_00128_TA", "sentence06")
    inference_loader = DataLoader(inference_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    gts = []
    preds = []
    with torch.no_grad():
        for i, data in track(enumerate(inference_loader), total=len(inference_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            gts.append(labels.reshape(-1, 5023, 3).cpu().numpy() / 10)
            preds.append(outputs.reshape(-1, 5023, 3).cpu().numpy() / 10)
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    gts = renderer.render(gts)
    preds = renderer.render(preds)
    images_to_video(gts, "gt.mp4")
    images_to_video(preds, "pred.mp4")


# inference()
if __name__ == "__main__":
    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader()
    )
    inference()
