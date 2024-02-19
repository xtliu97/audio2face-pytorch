import os
import pickle
from typing import TypedDict, List, Mapping, Literal, Tuple
from typing_extensions import Unpack
from copy import deepcopy
import dataclasses

import lmdb
import numpy as np
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from rich.progress import Progress, track
from rich import print


from .extractor import MelSpectrogramExtractor


def load_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def load_npy_mmapped(npy_path):
    return np.load(npy_path, mmap_mode="r")


VOCASET_AUDIO_TYPE = Mapping[str, Mapping[str, np.ndarray]]
VOCASET_VERTS_TYPE = np.ndarray
VOCASET_SUBJ_SEQ_TO_IDX_TYPE = Mapping[str, Mapping[str, Mapping[int, int]]]

training_subject = [
    "FaceTalk_170728_03272_TA",
    "FaceTalk_170904_00128_TA",
    "FaceTalk_170725_00137_TA",
    "FaceTalk_170915_00223_TA",
    "FaceTalk_170811_03274_TA",
    "FaceTalk_170913_03279_TA",
    "FaceTalk_170904_03276_TA",
    "FaceTalk_170912_03278_TA",
]
training_sentence = [f"sentence{i:02d}" for i in range(1, 41)]
validation_subject = [
    "FaceTalk_170811_03275_TA",
    "FaceTalk_170908_03277_TA",
]
validation_sentence = [f"sentence{i:02d}" for i in range(21, 41)]


@dataclasses.dataclass
class FrameData:
    human_id: str
    sentence_id: str
    seq_num: int
    audio: np.ndarray
    verts: np.ndarray
    feature: np.ndarray

    def __repr__(self) -> str:
        return f"""FrameData(
    human_id={self.human_id},
    sentence_id={self.sentence_id},
    seq_num={self.seq_num},
    audio: {self.audio.shape},
    verts: {self.verts.shape},
    feature: {self.feature.shape}
)    
"""


class AduioParams(TypedDict):
    fps: int
    sample_rate: int
    length: float


def get_audio_fragment(
    audio: np.ndarray, idx: int, **audio_params: Unpack[AduioParams]
) -> np.ndarray | None:
    n_samples = int(audio_params["sample_rate"] * audio_params["length"])
    pad_audio = np.concatenate([np.zeros(n_samples), audio, np.zeros(n_samples)])
    start = idx * audio_params["sample_rate"] // audio_params["fps"]
    end = start + n_samples
    # check if the audio is long enough
    if end > len(pad_audio):
        print(f"Audio is not long enough to get fragment: {end} > {len(pad_audio)}")
        return None
    return pad_audio[start:end]


class IndexDataset(Dataset):
    def open_db(self):
        if not hasattr(self, "env"):
            self.env = lmdb.open(self._db_path, map_size=1024 * 1024 * 1024 * 1024)
            if not self._write:
                self._cnt = self.env.stat()["entries"]

    def __init__(self, base_path, write=False):
        self.__init_failed = True  # for __del__
        db_path = os.path.join(base_path, "framedata")
        if write and os.path.exists(db_path):
            raise FileExistsError(f"DB {db_path} already exists")

        if not write and not os.path.exists(db_path):
            raise FileNotFoundError(f"DB {db_path} not found")

        self.__init_failed = False

        self._write = write
        self._db_path = db_path
        self._base_path = base_path
        self._cnt = 0

        if write:
            self._trainset_indexes = []
            self._valset_indexes = []
            self._testset_indexes = []
        else:
            self._trainset_indexes = self.__load_split_indices("train")
            self._valset_indexes = self.__load_split_indices("val")
            self._testset_indexes = self.__load_split_indices("test")

    @property
    def path(self):
        return self._base_path

    def ascii_key(self, key) -> bytes:
        return f"{key:08d}".encode("ascii")

    def insert_frame_data(self, frame_data: dict):
        self.open_db()
        frame_data = FrameData(**frame_data)
        assert self._write, "DB is not open for writing"
        with self.env.begin(write=True) as txn:
            txn.put(self.ascii_key(self._cnt), pickle.dumps(frame_data))
        if (
            frame_data.human_id in training_subject
            and frame_data.sentence_id in training_sentence
        ):
            self._trainset_indexes.append(
                f"{self._cnt} {frame_data.human_id} {frame_data.sentence_id}"
            )
        elif (
            frame_data.human_id in validation_subject
            and frame_data.sentence_id in validation_sentence
        ):
            self._valset_indexes.append(
                f"{self._cnt} {frame_data.human_id} {frame_data.sentence_id}"
            )
        else:
            self._testset_indexes.append(
                f"{self._cnt} {frame_data.human_id} {frame_data.sentence_id}"
            )
        self._cnt += 1

    def __save_split_indices(self, phase_name, indexes):
        filename = os.path.join(self._db_path, f"{phase_name}_splits.txt")
        with open(filename, "w") as f:
            f.write("\n".join(map(str, indexes)))
        print(f"Saved {phase_name} split indices to {filename}")

    def __load_split_indices(self, phase_name) -> List[int]:
        filename = os.path.join(self._db_path, f"{phase_name}_splits.txt")
        with open(filename, "r") as f:
            return list(map(str, f.read().split("\n")))

    def close(self):
        if self.__init_failed:
            return

        if self._write:
            self.__save_split_indices("train", self._trainset_indexes)
            self.__save_split_indices("val", self._valset_indexes)
            self.__save_split_indices("test", self._testset_indexes)
        if hasattr(self, "env"):
            self.env.close()

    def __del__(self):
        self.close()

    def __len__(self):
        return self.env.stat()["entries"]

    def __getitem__(self, index) -> FrameData:
        self.open_db()
        with self.env.begin() as txn:
            framedata = pickle.loads(txn.get(self.ascii_key(index)))
        return framedata

    def read_only(self):
        return IndexDataset(self.path, write=False)


def pre_fetch_length(
    data_verts: VOCASET_VERTS_TYPE,
    raw_audio: VOCASET_AUDIO_TYPE,
    subj_seq_to_idx: VOCASET_SUBJ_SEQ_TO_IDX_TYPE,
) -> int:
    n_all = 0
    for clip_name, clip_data in raw_audio.items():
        for sentence_id, audio_data in clip_data.items():
            if sentence_id not in subj_seq_to_idx[clip_name]:
                continue
            audio_idx = subj_seq_to_idx[clip_name][sentence_id]
            n_all += sum(len(audio_idx) for _ in audio_data)
    return n_all


def build_frame_data(
    data_verts: VOCASET_VERTS_TYPE,
    raw_audio: VOCASET_AUDIO_TYPE,
    subj_seq_to_idx: VOCASET_SUBJ_SEQ_TO_IDX_TYPE,
    deepspeech_feature,
    *,
    max_num: int | None = None,
    save_path: str | None = None,
) -> IndexDataset:
    # frame_data = []
    if save_path is None:
        save_path = os.path.dirname(__file__)
    dataset = IndexDataset(save_path, write=True)
    n_all, n_success = 0, 0
    total_audio_idx = pre_fetch_length(
        data_verts, raw_audio, subj_seq_to_idx
    )  # for progress bar
    T = MelSpectrogramExtractor(
        sample_rate=22000,
        n_mel=32,
        win_length=176 * 2,
        hop_length=176,
        n_fft=1024,
        normalize=True,
    )
    with Progress() as pbar:
        task = pbar.add_task("[green]Processing...", total=total_audio_idx)
        for i, clip_name, clip_data in zip(
            range(len(raw_audio)), raw_audio.keys(), raw_audio.values()
        ):
            pbar.update(
                task,
                description=f"Processing {clip_name} {i}/{len(raw_audio)}",
            )
            # print(f"Processing {clip_name} with {len(clip_data)} sentences")
            subtask = pbar.add_task(f"Processing {clip_name}", total=len(clip_data))
            if clip_name not in subj_seq_to_idx:
                print(f"Skipping {clip_name} as it is not in subj_seq_to_idx")
                n_all += len(clip_data)
                continue
            for sentence_id, audio_data in clip_data.items():
                if sentence_id not in subj_seq_to_idx[clip_name]:
                    print(
                        f"Skipping {sentence_id} as it is not in subj_seq_to_idx[{clip_name}]"
                    )
                    continue
                audio_idx = subj_seq_to_idx[clip_name][sentence_id]
                for idx, seq_num in audio_idx.items():
                    audio_frag = get_audio_fragment(
                        audio_data["audio"],
                        idx,
                        sample_rate=audio_data["sample_rate"],
                        length=0.52,
                        fps=60,
                    )
                    # feature = deepspeech_feature[clip_name][sentence_id]["audio"][idx]
                    feature = T(audio_frag)
                    n_all += 1
                    if audio_frag is None:
                        continue
                    dataset.insert_frame_data(
                        {
                            "human_id": clip_name,
                            "sentence_id": sentence_id,
                            "seq_num": seq_num,
                            "audio": audio_frag,
                            "verts": data_verts[seq_num],
                            "feature": feature,
                        }
                    )

                    n_success += 1
                    pbar.update(task, advance=1)
                pbar.update(subtask, advance=1)
            pbar.update(subtask, visible=False)
            if max_num and n_all >= max_num:
                break

    dataset.close()

    print(f"Built {n_success} frame data from {n_all} audio fragments")
    return dataset.read_only()


def build_dataset(src_datapath: str, dst_datapath: str = None) -> str:
    print("[bold green]Building dataset...")
    datapath = os.path.abspath(src_datapath)
    raw_audio_path = os.path.join(datapath, "raw_audio_fixed.pkl")
    data_verts_path = os.path.join(datapath, "data_verts.npy")
    subj_seq_to_idx_path = os.path.join(datapath, "subj_seq_to_idx.pkl")
    deepspeech_feature_path = os.path.join(datapath, "processed_audio_deepspeech.pkl")
    print(f"[bold green]Loaded data from {datapath}")
    #  data
    data_verts = load_npy_mmapped(data_verts_path)
    raw_audio = load_pickle(raw_audio_path)
    subj_seq_to_idx = load_pickle(subj_seq_to_idx_path)
    deepspeech_feature = load_pickle(deepspeech_feature_path)
    #  frame data
    frame_dataset = build_frame_data(
        data_verts,
        raw_audio,
        subj_seq_to_idx,
        deepspeech_feature,
        save_path=dst_datapath,
    )
    # print(f"Loaded {len(self._frame_data)} frame data")

    print(f"[bold green]Dataset saved to {frame_dataset.path}")
    return frame_dataset.path


class VocaSet(Dataset):
    def __init__(
        self, datapath: str, phase: Literal["train", "val", "test", "all"] = "all"
    ):
        self.datapath = os.path.abspath(datapath)
        self._frame_data = IndexDataset(self.datapath, write=False)
        self._phase = phase
        if phase == "all":
            self._indices = [
                *self._frame_data._trainset_indexes,
                *self._frame_data._valset_indexes,
                *self._frame_data._testset_indexes,
            ]
        elif phase == "train":
            self._indices = self._frame_data._trainset_indexes
        elif phase == "val":
            self._indices = self._frame_data._valset_indexes
        elif phase == "test":
            self._indices = self._frame_data._testset_indexes
        else:
            raise ValueError(f"Invalid phase {phase}")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        idx = int(self._indices[index].split()[0])
        return self._frame_data[idx]

    def get_framedatas(self, target_human_id: str, target_sentence_id: str):
        new_ins = VocaSet(self.datapath, phase="all")
        idxs = []
        for idx in self._indices:
            i, human_id, sentence_id = idx.split()
            if human_id == target_human_id and sentence_id == target_sentence_id:
                idxs.append(i)
        new_ins._indices = idxs
        return new_ins


def collate_fn(batch):
    inputs = []
    labels = []
    for sample in batch:
        inputs.append(sample.feature[:, :64].transpose(0, 1))
        labels.append(torch.tensor(sample.verts).reshape(-1).float())
    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    # normalize
    inputs = inputs * 10e3
    labels = labels * 10
    return inputs, labels


class VocaDataModule(pl.LightningDataModule):
    def __init__(self, datapath: str, batch_size: int = 32):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            VocaSet(self.datapath, phase="train"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            VocaSet(self.datapath, phase="val"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True,
        )
 
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            VocaSet(self.datapath, phase="test"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

