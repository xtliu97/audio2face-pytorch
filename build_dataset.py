import os
import pickle
from typing import TypedDict, List, Mapping, Literal, Tuple
from typing_extensions import Unpack
import dataclasses

from rich.progress import Progress, track
from rich import print

import lmdb
from cached_property import cached_property
import matplotlib.pyplot as plt
import librosa

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader


from dataset.vocaset import VocaSet, build_dataset


def check_dataset(dataset: VocaSet):
    for data in track(
        dataset, description=f"Checking dataset phase={dataset._phase}..."
    ):
        _ = data
    print(f"{len(dataset)} frame data of phase {dataset._phase} loaded")


if __name__ == "__main__":
    source_datapath = os.path.join(os.getcwd(), "../../Downloads/trainingdata/")
    dataset_path = os.getcwd()

    # build dataset
    build_dataset(source_datapath, dataset_path)

    # load dataset
    dataset = VocaSet(dataset_path)

    # random vis
    idx = np.random.randint(len(dataset))
    print(dataset[idx])
    fig, axs = plt.subplots(2, 1)

    # check dataset
    # check_dataset(dataset)
