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
    
    # Check if preprocessed data exists
    import os
    from core.config import CONFIG
    preprocessed_file = os.path.join(
        CONFIG.dataset_preprocessed_dir_path(), 
        "w2v2_text_face.pkl"
    )
    
    if not os.path.exists(preprocessed_file):
        logger.error(f"Preprocessed data not found: {preprocessed_file}")
        logger.error("You need to run preprocessing first:")
        logger.error("  python run_preprocessing.py")
        logger.error("")
        logger.error("OR if you just want to run the API server (no training needed):")
        logger.error("  python run_api.py")
        logger.error("  or: python api.py")
        return

    audio_trainer = TrainerOps.create_or_load_audio_trainer(
        checkpoint_name="wav2vec2_state_dict1.pt",
        load_state_dict=True,
        num_epochs=0,
        learning_rate=2e-5,
    )

    fusion_trainer = TrainerOps.create_or_load_fusion_trainer(
        audio_model=audio_trainer.model,
        checkpoint_name="fusion_state_dict.pt",
        load_state_dict=True,  # Changed to True - will load if exists
        num_epochs=5,
        learning_rate=1e-4,
        freeze_audio=True,
    )

    # Only train if checkpoint doesn't exist or you want to retrain
    fusion_checkpoint = os.path.join(CONFIG.saved_models_location(), "fusion_state_dict.pt")
    if not os.path.exists(fusion_checkpoint):
        logger.info("Fusion checkpoint not found. Starting training...")
        TrainerOps.train(fusion_trainer)
        TrainerOps.save(fusion_trainer, "fusion_state_dict.pt", save_state_dict=True)
    else:
        logger.info("Fusion checkpoint found. Skipping training. Use test_models.py to evaluate.")

    TrainerOps.save(fusion_trainer, "fusion_state_dict.pt", save_state_dict=True)

    TrainerOps.evaluate(fusion_trainer)


if __name__ == "__main__":
    main()
