import os

from rich.progress import track
from rich import print

import numpy as np
import matplotlib.pyplot as plt

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
    # build_dataset(source_datapath, dataset_path)

    # load dataset
    dataset = VocaSet(dataset_path)
    print(len(dataset))

    # random vis
    for i, data in enumerate(dataset):
        if i == 1000:
            print(data)
            print(data.verts)
            print(data.feature)
            break

    # get frame data
    frames = dataset.get_framedatas("FaceTalk_170904_00128_TA", "sentence06")
    print(len(frames))
    

    # check dataset
    # check_dataset(dataset)
