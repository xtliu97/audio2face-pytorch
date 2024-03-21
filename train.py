import os

import torch
import argparse
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar as ProgressBar,
    # DeviceStatsMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.dataset.vocaset import VocaDataModule
from src.model.lightning_model import Audio2FaceModel, ExpConfig


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    torch.set_float32_matmul_precision("medium")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # training parameters
    dataset_path = os.getcwd() + "/.."
    config = ExpConfig.from_yaml(args.config)

    is_transformer = config.modelname == "faceformer"
    if is_transformer:
        config.split_frame = False
        config.batch_size = 1
        config.feature_extractor = None

    voca_datamodule = VocaDataModule(
        dataset_path,
        batch_size=config.batch_size,
        num_workers=8,
        split_frame=config.split_frame,
    )

    version = config.name()

    # Train
    model = Audio2FaceModel(config)

    trainer = L.Trainer(
        precision=config.percision,
        log_every_n_steps=10,
        logger=TensorBoardLogger("logs", name=version),
        callbacks=[
            ModelCheckpoint(monitor="val/err", save_last=False),
            EarlyStopping(monitor="val/err", patience=5),
            ProgressBar(),
            # DeviceStatsMonitor(),
        ],
        max_epochs=50,
    )
    trainer.fit(model, datamodule=voca_datamodule)

    ckpts = os.listdir(trainer.log_dir + "/checkpoints")
    sorted_ckpts = sorted(ckpts, key=lambda x: int(x.split("=")[-1].split(".")[0]))

    model = Audio2FaceModel.load_from_checkpoint(
        trainer.log_dir + "/checkpoints/" + sorted_ckpts[-1]
    )

    # inference only
    # trainer = L.Trainer()
    # voca_datamodule.setup("test")

    trainer.predict(
        model,
        voca_datamodule.predict_dataloader("FaceTalk_170908_03277_TA", "sentence02"),
    )
