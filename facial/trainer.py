import logging

import torch
import torch.nn as nn

from core.config import device
from core.trainer import AbstractTrainer

logger = logging.getLogger(__name__)


class FacialTrainer(AbstractTrainer):
    def __init__(
        self,
        facial_model: nn.Module,
        num_epochs: int = 15,
        learning_rate: float = 1e-3,
        loss: callable = None,
    ):
        if loss is None:
            loss = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            facial_model.parameters(),
            lr=learning_rate,
            eps=1e-8,
        )

        super().__init__(
            facial_model,
            num_epochs,
            optimizer,
            loss,
            name="facial",
        )

    def _get_logits_and_real(self, batch):
        _, face, emotion = batch

        face = face.to(device)
        emotion = emotion.to(device)

        logits = self.model(face)

        return logits, emotion
