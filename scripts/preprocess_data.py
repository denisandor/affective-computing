import logging
import os
from typing import Any
import torch
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from core.config import CONFIG
from preprocessing.iemocap import IemocapPreprocessor

logger = logging.getLogger(__name__)


def _ensure_preprocessed_dir() -> str:
    preprocessed_dir = CONFIG.dataset_preprocessed_dir_path()
    os.makedirs(preprocessed_dir, exist_ok=True)
    return preprocessed_dir


def _preprocessed_path(filename: str) -> str:
    return os.path.join(_ensure_preprocessed_dir(), filename)


def process_raw_data_to_pickle(output_filename: str) -> None:
    dataset_path = CONFIG.dataset_path()
    logger.info(f"Building IEMOCAP dataframe with faces from: {dataset_path}")

    preprocessor = IemocapPreprocessor(dataset_path)
    df = preprocessor.generate_dataframe()

    out_path = _preprocessed_path(output_filename)
    df.to_pickle(out_path)
    logger.info(f"Saved dataframe with face tensors to: {out_path}")
    logger.info(f"Dataframe shape: {df.shape}")
    logger.info(f"Valid face samples: {sum(1 for x in df['face'] if isinstance(x, torch.Tensor))}")


def process_audio_data_to_pickle(
        input_filename: str,
        output_filename: str,
        audio_extractor: Any,
) -> None:
    in_path = _preprocessed_path(input_filename)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input pickle not found: {in_path}")

    df = pd.read_pickle(in_path)
    dataset_path = CONFIG.dataset_path()

    logger.info(f"Extracting audio features for {len(df)} samples")

    audio_features = []
    for idx, row in df.iterrows():
        utterance_id = row["audio"]
        session_num = int(utterance_id[3:5])
        session_dir = f"Session{session_num}"
        parent_folder = "_".join(utterance_id.split("_")[:-1])
        wav_path = os.path.join(
            dataset_path, session_dir, "sentences", "wav", parent_folder, utterance_id + ".wav"
        )
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Could not find wav file: {wav_path}")
        features = audio_extractor.extract(wav_path)
        audio_features.append(features)

    df["audio"] = audio_features
    out_path = _preprocessed_path(output_filename)
    df.to_pickle(out_path)
    logger.info(f"Saved with audio features: {out_path}")


def process_facial_data_to_pickle(input_filename: str, output_filename: str) -> None:
    logger.warning("process_facial_data_to_pickle() is deprecated. "
                   "Use updated process_raw_data_to_pickle() instead which includes faces.")


def preprocess_pipeline(audio_extractor=None):
    raw_file = "iemocap_raw.pkl"
    audio_file = "w2v2_audio_face.pkl"
    final_file = "w2v2_text_face.pkl"  # Changed to match dataloader expectation

    process_raw_data_to_pickle(raw_file)

    if audio_extractor:
        process_audio_data_to_pickle(raw_file, audio_file, audio_extractor)

        df = pd.read_pickle(_preprocessed_path(audio_file))
        df.to_pickle(_preprocessed_path(final_file))
        logger.info(f"Complete pipeline done: {_preprocessed_path(final_file)}")
    else:
        logger.info(f"Raw data with faces ready: {_preprocessed_path(raw_file)}")

