import os

from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    # DeviceStatsMonitor,
)

from dataset.vocaset import VocaDataModule
from model.audio2face import Audio2Face

# %% load dataset
EXPNAME = "shuffle_1_10_randomshift_5"

# %% trainer
batch_size = 512

if __name__ == "__main__":
    train_batch_size = 512
    val_batch_size = 256

    # torch.multiprocessing.set_start_method("spawn")
    # torch.set_float32_matmul_precision("medium")

    dataset_path = os.getcwd() + "/.."
    voca_datamodule = VocaDataModule(
        dataset_path,
        batch_size=batch_size,
        num_workers=8,
    )

    model = Audio2Face(5023 * 3, 12)
    trainer = L.Trainer(
        callbacks=[
            ModelCheckpoint(monitor="val/err", save_last=True),
            EarlyStopping(monitor="val/err", patience=5),
            RichProgressBar(),
            # DeviceStatsMonitor(),
        ],
        max_epochs=50,
    )
    trainer.fit(model, datamodule=voca_datamodule)

    model = Audio2Face.load_from_checkpoint(f"{trainer.log_dir}/checkpoints/last.ckpt")
    trainer = L.Trainer(logger=False)
    trainer.predict(
        model,
        voca_datamodule.predict_dataloader("FaceTalk_170908_03277_TA", "sentence02"),
    )
