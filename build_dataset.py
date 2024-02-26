import os

from rich.progress import track
from rich import print

from dataset.vocaset import VocaSet, build_dataset


def check_dataset(dataset: VocaSet):
    for data in track(
        dataset, description=f"Checking dataset phase={dataset._phase}..."
    ):
        _ = data
    print(f"{len(dataset)} frame data of phase {dataset._phase} loaded")


if __name__ == "__main__":
    source_datapath = os.path.join(os.getcwd(), "../")
    dataset_path = os.getcwd()

    # build dataset
    build_dataset(source_datapath, dataset_path)

    # load dataset
    dataset = VocaSet(dataset_path)

    # random vis
    for data in dataset:
        print(data)
        break

    # get frame data
    frames = dataset.get_framedatas("FaceTalk_170725_00137_TA", "sentence28")
    print(len(frames))

    # check dataset
    # check_dataset(dataset)
