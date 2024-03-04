import os

from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
    # DeviceStatsMonitor,
)

from dataset.vocaset import ClipVocaSet
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
    trainset = ClipVocaSet(dataset_path, phase="train")
    valset = ClipVocaSet(dataset_path, phase="val")
    train_loader = DataLoader(
        trainset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=12,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=12,
        persistent_workers=True,
    )

    testset = ClipVocaSet(dataset_path, phase="all").get_framedatas(
        "FaceTalk_170908_03277_TA", "sentence02"
    )
    testlaoder = DataLoader(
        testset,
        batch_size=20,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Trainset size: {len(trainset)}")
    print(f"Valset size: {len(valset)}")
    print(f"Testset size: {len(testset)}")

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
    trainer.fit(model, train_loader, val_loader)

    model = Audio2Face.load_from_checkpoint(f"{trainer.log_dir}/checkpoints/last.ckpt")
    trainer = L.Trainer(logger=False)
    trainer.predict(model, testlaoder)
