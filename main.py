import logging
import random

import numpy as np
import torch

from core.config import CONFIG
from scripts.run_model import TrainerOps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_env():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    CONFIG.load_config("config.yaml")


def main():
    prepare_env()

    audio_trainer = TrainerOps.create_or_load_audio_trainer(
        checkpoint_name="wav2vec2_state_dict1.pt",
        load_state_dict=True,
        num_epochs=0,
        learning_rate=2e-5,
    )

    fusion_trainer = TrainerOps.create_or_load_fusion_trainer(
        audio_model=audio_trainer.model,
        checkpoint_name="fusion_state_dict.pt",
        load_state_dict=False,
        num_epochs=5,
        learning_rate=1e-4,
        freeze_audio=True,
    )

    TrainerOps.train(fusion_trainer)

    TrainerOps.save(fusion_trainer, "fusion_state_dict.pt", save_state_dict=True)

    TrainerOps.evaluate(fusion_trainer)


if __name__ == "__main__":
    main()
