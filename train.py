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
    batch_size = 128
    modelname = "af_model"
    vertex_count = 5023 * 3
    one_hot_size = 12
    split_frame = True
    percision = "16-mixed"
    lr = 1e-4
    feature_extractor = "wav2vec"
    sample_rate = 22000
    n_feature = 32
    out_dim = 52
    win_length = 220 * 2

    is_transformer = modelname == "faceformer"
    if is_transformer:
        split_frame = False
        batch_size = 1
        feature_extractor = None

    voca_datamodule = VocaDataModule(
        dataset_path,
        batch_size=batch_size,
        num_workers=8,
        split_frame=split_frame,
    )

    version = f"{modelname}/{feature_extractor}-{lr}"

    # Train
    model = Audio2FaceModel(
        modelname,
        feature_extractor,
        vertex_count,
        one_hot_size,
        n_feature=n_feature,
        out_dim=out_dim,
        win_length=win_length,
        lr=lr,
    )

    trainer = L.Trainer(
        precision=percision,
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
