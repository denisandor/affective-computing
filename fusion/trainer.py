import torch
from torch import nn
from core.config import device
from core.trainer import AbstractTrainer


class FusionTrainer(AbstractTrainer):
    def __init__(
        self,
        fusion_model: nn.Module,
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
        loss: callable = nn.CrossEntropyLoss(),
    ):
        optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=learning_rate)
        super().__init__(
            fusion_model,
            num_epochs,
            optimizer,
            loss,
            name="fusion",
        )

    def _get_logits_and_real(self, batch):
        audio, face, emotion = batch

        audio = audio.to(device)

        if isinstance(face, list):
            face = torch.stack(face, dim=0)

        face = face.to(device).float()
        emotion = emotion.to(device)

        logits = self.model(audio, face)
        return logits, emotion
