import os

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar as ProgressBar,
    # DeviceStatsMonitor,
)

from src.dataset.vocaset import VocaDataModule
from src.model.lightning_model import Audio2FaceModel


if __name__ == "__main__":
    train_batch_size = 512
    val_batch_size = 256

    # torch.multiprocessing.set_start_method("spawn")
    # torch.set_float32_matmul_precision("medium")

    dataset_path = os.getcwd() + "/.."
    voca_datamodule = VocaDataModule(
        dataset_path,
        batch_size=train_batch_size,
        num_workers=8,
    )

    model = Audio2FaceModel("audio2mesh", 5023 * 3, 12)
    trainer = L.Trainer(
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
    )

    # inference only
    # trainer = L.Trainer()
    # voca_datamodule.setup("test")

    trainer.predict(
        model,
        voca_datamodule.predict_dataloader("FaceTalk_170908_03277_TA", "sentence02"),
    )
