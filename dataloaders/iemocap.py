import logging
import numpy as np
import pandas as pd
from typing import List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


class IemocapDataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            dataframe: pd.DataFrame,
            emotions: List[str],
            split: str,
    ):
        self._dataset_path = dataset_path
        self._emotions = np.array(emotions)
        self._dataframe = dataframe
        self._split = split

        self._dataframe = self._dataframe.loc[self._dataframe["emotion"].isin(emotions)].copy()

        self._setup_splits()

        self._filter_valid_faces()

        self._setup_transforms()

        logger.info(f"Loaded {split} dataset. Size: {len(self)}")
        self._log_emotion_stats()

    def _setup_splits(self):
        if 'session' not in self._dataframe.columns:
            rows_80_percent = int(0.8 * len(self._dataframe))
            if self._split == "train":
                self._dataframe = self._dataframe.iloc[:rows_80_percent, :]
            elif self._split == "val" or self._split == "test":
                self._dataframe = self._dataframe.iloc[rows_80_percent:, :]
            return

        session_to_fold = {'Ses01': 0, 'Ses02': 0, 'Ses03': 1, 'Ses04': 1, 'Ses05': 2}
        if self._split == "train":
            self._dataframe = self._dataframe[
                ~self._dataframe['session'].str.contains('Ses05', na=False)
            ]
        elif self._split == "val":
            self._dataframe = self._dataframe[
                self._dataframe['session'].str.contains('Ses05', na=False)
            ]
        elif self._split == "test":
            self._dataframe = self._dataframe[
                self._dataframe['session'].str.contains('Ses04', na=False)
            ]

    def _filter_valid_faces(self):
        if "face" not in self._dataframe.columns:
            return

        before = len(self._dataframe)

        def has_valid_face(row):
            face = row["face"]
            return isinstance(face, torch.Tensor) and face.numel() > 0 and not torch.all(face == 0)

        self._dataframe = self._dataframe[self._dataframe.apply(has_valid_face, axis=1)]
        after = len(self._dataframe)
        logger.info(f"Filtered invalid faces in {self._split} split: {before} -> {after}")

    def _setup_transforms(self):
        if "face" not in self._dataframe.columns:
            self.transform = None
            return

        if self._split == "train":
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _log_emotion_stats(self):
        emotions_str = ""
        for emotion in self._emotions:
            count = self._dataframe[self._dataframe['emotion'] == emotion]['emotion'].count()
            emotions_str += f"{emotion}: {count} | "
        logger.info(f"{self._split} emotion counts: {emotions_str[:-3]}")

        self.class_weights = 1.0 / self._dataframe['emotion'].value_counts().reindex(self._emotions, fill_value=1)
        logger.info(f"Class weights: {self.class_weights.to_dict()}")

    def preprocess_face(self, face_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if not isinstance(face_tensor, torch.Tensor) or torch.all(face_tensor == 0):
            return None

        if face_tensor.dim() == 3 and face_tensor.shape[0] in [1, 3]:
            face_tensor = torch.clamp(face_tensor.float(), 0, 1)
            if self.transform:
                return self.transform(face_tensor)
        return None

    def __getitem__(self, index: int):
        row = self._dataframe.iloc[index]
        audio = row["audio"]
        emotion = row["emotion"]

        if "face" in self._dataframe.columns:
            face_raw = row["face"]
            face = self.preprocess_face(face_raw)
            if face is None:
                face = torch.zeros(3, 224, 224)
        else:
            face = row.get("text", None)

        emotion_index = torch.tensor(np.where(self._emotions == emotion)[0][0])
        return audio, face, emotion_index

    def __len__(self):
        return len(self._dataframe)

    @property
    def class_weights_tensor(self):
        return torch.tensor(self.class_weights.values, dtype=torch.float32)


def iemocap_collate_fn(batch):
    audios, faces, labels = zip(*batch)

    lengths = [a.shape[0] for a in audios]
    max_len = max(lengths)
    padded_audios = []
    for a in audios:
        if a.shape[0] < max_len:
            pad_len = max_len - a.shape[0]
            padded = torch.cat([a, torch.zeros(pad_len, dtype=a.dtype, device=a.device)])
        else:
            padded = a[:max_len]
        padded_audios.append(padded)
    padded_audios = torch.stack(padded_audios, dim=0)

    faces = torch.stack(faces, dim=0)
    labels = torch.stack(labels, dim=0)

    return padded_audios, faces, labels


def IemocapDataLoader(
        dataset_path: str,
        dataframe: pd.DataFrame,
        emotions: List[str],
        split: str,
        **kwargs,
):
    dataset = IemocapDataset(dataset_path, dataframe, emotions, split)
    return DataLoader(
        dataset,
        collate_fn=iemocap_collate_fn,
        **kwargs,
    )
