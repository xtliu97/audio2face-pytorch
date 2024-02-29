import os
import pickle
from typing import TypedDict, Mapping, Literal, cast
from typing_extensions import Unpack
import dataclasses
from functools import lru_cache

import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset
from rich import print
from rich.progress import Progress


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
test_subject = ["FaceTalk_170809_00138_TA", "FaceTalk_170731_00024_TA"]


def get_human_id_one_hot(human_id: str):
    all_human_id = [*training_subject, *validation_subject, *test_subject]
    one_hot = np.zeros(len(all_human_id))
    one_hot[all_human_id.index(human_id)] = 1
    return one_hot


@lru_cache(maxsize=20)
def get_template_vert(datapath, human_id):
    template_verts_path = os.path.join(datapath, "templates.pkl")
    template_verts = load_pickle(template_verts_path)
    return template_verts[human_id]


@dataclasses.dataclass
class DataFrame:
    human_id: str
    sentence_id: str
    audio: np.ndarray
    sample_rate: int
    verts: np.ndarray
    # tempalte_vert: np.ndarray
    fps: int

    def __repr__(self) -> str:
        return f"""DataFrame(
    human_id={self.human_id},
    sentence_id={self.sentence_id},
    audio: {self.audio.shape},
    verts: {self.verts.shape},
    tempalte_vert: {self.tempalte_vert.shape}
    fps: {self.fps}
)"""


class VocaItem(TypedDict):
    audio: torch.Tensor
    verts: torch.Tensor
    template_vert: torch.Tensor
    one_hot: torch.Tensor
    feature: torch.Tensor


def build_data_frames(
    data_verts: VOCASET_VERTS_TYPE,
    raw_audio: VOCASET_AUDIO_TYPE,
    subj_seq_to_idx: VOCASET_SUBJ_SEQ_TO_IDX_TYPE,
    template_verts,
    *,
    save_path: str | None = None,
) -> None:
    save_path = os.path.join(save_path, "clipdata")
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "clipdata")

    os.makedirs(save_path, exist_ok=False)

    env = lmdb.open(save_path, map_size=1024**4, lock=False)

    global_idx = 0

    with Progress() as pbar:
        task = pbar.add_task(
            "[green]Processing...",
            total=pre_fetch_length(data_verts, raw_audio, subj_seq_to_idx),
        )
        for i, (clip_name, clip_data) in enumerate(raw_audio.items()):
            if clip_name not in subj_seq_to_idx:
                print(f"Skipping {clip_name} as it is not in subj_seq_to_idx")
                continue
            for j, (sentence_id, audio_data) in enumerate(clip_data.items()):
                pbar.update(
                    task,
                    description=f"Processing {clip_name} {sentence_id} {j}/{len(clip_data)}",
                )
                if sentence_id not in subj_seq_to_idx[clip_name]:
                    print(
                        f"Skipping {sentence_id} as it is not in subj_seq_to_idx[{clip_name}]"
                    )
                    continue
                audio_idxs = subj_seq_to_idx[clip_name][sentence_id]
                # verts = []
                for idx, seq_num in audio_idxs.items():
                    # verts.append(data_verts[seq_num])
                    # verts = np.stack(verts)
                    data_clip = DataFrame(
                        human_id=clip_name,
                        sentence_id=sentence_id,
                        audio=get_audio_fragment(
                            audio_data["audio"],
                            idx,
                            sample_rate=audio_data["sample_rate"],
                            length=0.52,
                            fps=60,
                        ),
                        sample_rate=audio_data["sample_rate"],
                        verts=data_verts[seq_num],
                        # tempalte_vert=template_verts[clip_name],
                        fps=60,
                    )
                    with env.begin(write=True) as txn:
                        txn.put(str(global_idx).encode(), pickle.dumps(data_clip))
                    global_idx += 1
                    pbar.update(task, advance=1)
    env.close()

    # save template verts
    with open(os.path.join(save_path, "templates.pkl"), "wb") as f:
        pickle.dump(template_verts, f)


class ClipVocaSet(Dataset):
    def __init__(
        self,
        datapath: str,
        phase: Literal["train", "val", "test", "all"] = "all",
        device="cpu",
    ):
        self.datapath = os.path.join(datapath, "clipdata")
        self.tempalte_verts = load_pickle(os.path.join(self.datapath, "templates.pkl"))
        self.phase = phase
        self.device = device
        self.setup()
        if phase == "train":
            self.datalist = self.trainlist
        elif phase == "val":
            self.datalist = self.vallist
        elif phase == "test":
            self.datalist = self.testlist
        elif phase == "all":
            self.datalist = [*self.trainlist, *self.vallist, *self.testlist]

    def setup(self):
        self.trainlist = []
        self.vallist = []
        self.testlist = []
        # human_id, sentence_id = file.rsplit("_", 1)  # same
        env = lmdb.open(self.datapath, map_size=1024**4, lock=False)
        txn = env.begin()
        for key, value in txn.cursor():
            key = key.decode("ascii")
            dataclip = pickle.loads(value)
            dataclip = cast(DataFrame, dataclip)
            if (
                dataclip.human_id in training_subject
                and dataclip.sentence_id in training_sentence
            ):
                self.trainlist.append(key)
            elif (
                dataclip.human_id in validation_subject
                and dataclip.sentence_id in validation_sentence
            ):
                self.vallist.append(key)
            else:
                self.testlist.append(key)
        print(f"Loaded {len(self.trainlist)} training clips")
        print(f"Loaded {len(self.vallist)} validation clips")
        print(f"Loaded {len(self.testlist)} test clips")

    def __init_env(self):
        if not hasattr(self, "_env"):
            self._env = lmdb.open(self.datapath, map_size=1024**4, lock=False)
            self._txn = self._env.begin()

    def get_single_item(self, key) -> VocaItem:
        self.__init_env()
        tmp = self._txn.get(key.encode())
        tmp = cast(DataFrame, pickle.loads(tmp))
        return VocaItem(
            audio=torch.FloatTensor(tmp.audio),
            verts=torch.FloatTensor(tmp.verts),
            template_vert=torch.FloatTensor(self.tempalte_verts[tmp.human_id]),
            one_hot=torch.FloatTensor(get_human_id_one_hot(tmp.human_id)),
        )

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.get_single_item(self.datalist[idx])

    def get_framedatas(self, target_human_id: str, target_sentence_id: str):
        res = []
        env = lmdb.open(self.datapath, map_size=1024**4, lock=False)
        txn = env.begin()
        for key, value in txn.cursor():
            dataclip = pickle.loads(value)
            dataclip = cast(DataFrame, dataclip)
            if (
                dataclip.human_id == target_human_id
                and dataclip.sentence_id == target_sentence_id
            ):
                res.append(self.get_single_item(key.decode()))

        return res


class AduioParams(TypedDict):
    fps: int
    sample_rate: int
    length: float


def get_audio_fragment(
    audio: np.ndarray, idx: int, **audio_params: Unpack[AduioParams]
) -> np.ndarray | None:
    n_pad_samples = int(audio_params["sample_rate"] * audio_params["length"] / 2)
    pad_audio = np.concatenate(
        [np.zeros(n_pad_samples), audio, np.zeros(n_pad_samples)]
    )
    start = idx * audio_params["sample_rate"] // audio_params["fps"]
    end = start + n_pad_samples * 2
    # check if the audio is long enough
    if end > len(pad_audio):
        print(f"Audio is not long enough to get fragment: {end} > {len(pad_audio)}")
        return None
    return pad_audio[start:end]


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
            n_all += len(audio_idx)
    return n_all


def build_dataset(src_datapath: str, dst_datapath: str = None) -> str:
    print("[bold green]Building dataset...")
    datapath = os.path.abspath(src_datapath)
    raw_audio_path = os.path.join(datapath, "raw_audio_fixed.pkl")
    data_verts_path = os.path.join(datapath, "data_verts.npy")
    subj_seq_to_idx_path = os.path.join(datapath, "subj_seq_to_idx.pkl")
    # deepspeech_feature_path = os.path.join(datapath, "processed_audio_deepspeech.pkl")
    template_verts_path = os.path.join(datapath, "templates.pkl")
    print(f"[bold green]Loaded data from {datapath}")
    #  data
    data_verts = load_npy_mmapped(data_verts_path)
    raw_audio = load_pickle(raw_audio_path)
    subj_seq_to_idx = load_pickle(subj_seq_to_idx_path)
    # deepspeech_feature = load_pickle(deepspeech_feature_path)
    template_verts = load_pickle(template_verts_path)
    #  frame data
    build_data_frames(
        data_verts,
        raw_audio,
        subj_seq_to_idx,
        # deepspeech_feature,
        save_path=dst_datapath,
        template_verts=template_verts,
    )
    # print(f"Loaded {len(self._frame_data)} frame data")

    print(f"[bold green]Dataset saved to {dst_datapath}")


if __name__ == "__main__":
    ClipVocaSet("clipdata")
