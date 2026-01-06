import os.path
import pandas as pd
import torch

from core.config import CONFIG
from dataloaders.iemocap import IemocapDataLoader


def get_dataloader(
    split=None,
    shuffle=None,
    mode: str = "audio",
):
    if split is None:
        split = ["train", "test"]
    if shuffle is None:
        shuffle = [True, False]

    pkl_name = "w2v2_text_face.pkl"

    dataframe = pd.read_pickle(
        os.path.join(CONFIG.dataset_preprocessed_dir_path(), pkl_name)
    )

    if mode == "face":
        if "face" not in dataframe.columns:
            raise ValueError("No 'face' column found. Run updated preprocessor first.")
        dataframe = dataframe[dataframe["face"].apply(
            lambda x: isinstance(x, torch.Tensor) and not torch.all(x == 0)
        )].copy()
    elif mode == "audio":
        pass
    elif mode == "multimodal":
        if "face" not in dataframe.columns:
            raise ValueError("No 'face' column. Update preprocessor.")
        dataframe = dataframe[dataframe["face"].apply(
            lambda x: isinstance(x, torch.Tensor) and not torch.all(x == 0)
        )].copy()
    else:
        raise ValueError("mode must be 'audio', 'face', or 'multimodal'")

    print(f"Mode: {mode}, Dataset size after filtering: {len(dataframe)}")

    if isinstance(split, str):
        return IemocapDataLoader(
            CONFIG.dataset_path(),
            dataframe,
            CONFIG.dataset_emotions(),
            split,
            **CONFIG.dataloader_dict(),
            shuffle=shuffle,
        )
    else:
        return [
            IemocapDataLoader(
                CONFIG.dataset_path(),
                dataframe,
                CONFIG.dataset_emotions(),
                split_item,
                **CONFIG.dataloader_dict(),
                shuffle=shuffle_item,
            )
            for split_item, shuffle_item in zip(split, shuffle)
        ]
