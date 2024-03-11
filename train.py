import os

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar as ProgressBar,
    # DeviceStatsMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.dataset.vocaset import VocaDataModule
from src.model.lightning_model import Audio2FaceModel


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    torch.set_float32_matmul_precision("medium")

    # training parameters
    dataset_path = os.getcwd() + "/.."
    batch_size = 64
    modelname = "audio2mesh"
    vertex_count = 5023 * 3
    one_hot_size = 12
    split_frame = True
    percision = "16-mixed"

    is_transformer = modelname == "faceformer"
    if is_transformer:
        split_frame = False
        batch_size = 1

    voca_datamodule = VocaDataModule(
        dataset_path,
        batch_size=batch_size,
        num_workers=8,
        split_frame=split_frame,
    )

    # Train
    model = Audio2FaceModel(modelname, vertex_count, one_hot_size)
    trainer = L.Trainer(
        precision=percision,
        logger=TensorBoardLogger("logs", name=modelname),
        callbacks=[
            ModelCheckpoint(monitor="val/err", save_last=True),
            EarlyStopping(monitor="val/err", patience=5),
            ProgressBar(),
            # DeviceStatsMonitor(),
        ],
        max_epochs=50,
    )
    trainer.fit(model, datamodule=voca_datamodule)

    model = Audio2FaceModel.load_from_checkpoint(
        f"{trainer.log_dir}/checkpoints/last.ckpt"
        # "/home/lixiang/lxt/VOCA-Pytorch/logs/faceformer/version_1/checkpoints/epoch=5-step=1884.ckpt"
    )

    # inference only
    # trainer = L.Trainer()
    # voca_datamodule.setup("test")

    trainer.predict(
        model,
        voca_datamodule.predict_dataloader("FaceTalk_170908_03277_TA", "sentence02"),
    )
