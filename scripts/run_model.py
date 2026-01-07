import logging
import os

import torch
from torch import nn

from core.config import CONFIG, device
from scripts.get_dataloaders import get_dataloader

from typing import Optional
from audio.wav2vec2 import Wav2Vec2
from audio.trainer import AudioTrainer

from fusion.model import FusionModel
from fusion.trainer import FusionTrainer

logger = logging.getLogger(__name__)


class TrainerOps:
    @staticmethod
    def _get_preprocessed_path(filename: str) -> str:
        return os.path.join(CONFIG.dataset_preprocessed_dir_path(), filename)

    @staticmethod
    def _get_dataloaders(mode: str):

        if mode == "audio":
            return get_dataloader(split=["train", "test"], mode="audio")
        elif mode == "fusion":
            return get_dataloader(split=["train", "test"], mode="multimodal")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def create_or_load_audio_trainer(
        checkpoint_name: str,
        load_state_dict: bool = False,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
    ) -> AudioTrainer:

        num_classes = len(CONFIG.dataset_emotions())
        model = Wav2Vec2(num_classes=num_classes)

        if load_state_dict:
            ckpt_path = os.path.join(CONFIG.saved_models_location(), checkpoint_name)
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded audio checkpoint: {ckpt_path}")
            else:
                logger.warning(f"Audio checkpoint not found: {ckpt_path}")

        trainer = AudioTrainer(
            audio_model=model,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )
        return trainer

    @staticmethod
    def create_or_load_fusion_trainer(
            audio_model: nn.Module,
            checkpoint_name: Optional[str] = None,
            load_state_dict: bool = False,
            num_epochs: int = 5,
            learning_rate: float = 1e-4,
            freeze_audio: bool = True,
            face_dim: int = 7,
    ) -> FusionTrainer:
        num_classes = len(CONFIG.dataset_emotions())

        fusion_model = FusionModel(
            num_classes=num_classes,
            wav2vec2_model=audio_model,
            face_dim=face_dim,
            freeze_audio=freeze_audio,
        )

        if load_state_dict and checkpoint_name is not None:
            ckpt_path = os.path.join(CONFIG.saved_models_location(), checkpoint_name)
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location=device)
                fusion_model.load_state_dict(state_dict)
                logger.info(f"Loaded fusion checkpoint: {ckpt_path}")
            else:
                logger.warning(f"Fusion checkpoint not found: {ckpt_path}")

        trainer = FusionTrainer(
            fusion_model=fusion_model,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
        )
        return trainer

    @staticmethod
    def train(trainer):
        mode = trainer._name
        train_loader, _ = TrainerOps._get_dataloaders(mode)
        logger.info(f"Training {mode} (samples: {len(train_loader.dataset)})")
        trainer.train(train_loader)

    @staticmethod
    def evaluate(trainer):
        mode = trainer._name
        _, test_loader = TrainerOps._get_dataloaders(mode)
        logger.info(f"Evaluating {mode} (test samples: {len(test_loader.dataset)})")
        trainer.eval(test_loader, labels=CONFIG.dataset_emotions())

    @staticmethod
    def save(trainer, checkpoint_name: str, save_state_dict: bool = True):
        save_dir = CONFIG.saved_models_location()
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, checkpoint_name)

        model = trainer.model
        if save_state_dict:
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved state_dict ({trainer._name}): {ckpt_path}")
        else:
            torch.save(model, ckpt_path)
            logger.info(f"Saved full model ({trainer._name}): {ckpt_path}")
