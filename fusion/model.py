import torch
import torchvision.models as models
from torch import nn
from audio.wav2vec2 import Wav2Vec2


class FusionModel(nn.Module):
    def __init__(
            self,
            num_classes: int,
            wav2vec2_model: Wav2Vec2,
            face_dim: int = 512,
            audio_proj_dim: int = 256,
            hidden_layers=None,
            freeze_audio: bool = True,
            freeze_face: bool = True,
            dropout: float = 0.3,
            load_pretrained_face: bool = True,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [512, 256, 128]

        self.audio_model = wav2vec2_model
        self.num_classes = num_classes
        self.face_dim = face_dim

        if freeze_audio:
            for p in self.audio_model.wav2vec2.parameters():
                p.requires_grad = False

        audio_hidden_dim = 768
        self.audio_proj = nn.Linear(audio_hidden_dim, audio_proj_dim)

        self.face_backbone = models.resnet18(weights='IMAGENET1K_V1' if load_pretrained_face else None)
        self.face_backbone.fc = nn.Identity()

        if freeze_face:
            for p in self.face_backbone.parameters():
                p.requires_grad = False

        self.face_proj = nn.Linear(512, face_dim)

        fusion_input_dim = audio_proj_dim + face_dim
        layers = []
        in_dim = fusion_input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        self.mlp = nn.Sequential(*layers)
        self.cls_head = nn.Linear(in_dim, num_classes)

    def forward(self, audio: torch.Tensor, face: torch.Tensor):
        batch_size = audio.size(0)

        outputs = self.audio_model.wav2vec2(audio)
        last_hidden = outputs.last_hidden_state
        audio_pooled = last_hidden.mean(dim=1)
        audio_emb = self.audio_proj(audio_pooled)

        face_features = self.face_backbone(face)
        face_emb = self.face_proj(face_features)

        fusion_in = torch.cat([audio_emb, face_emb], dim=1)
        h = self.mlp(fusion_in)
        logits = self.cls_head(h)
        return logits

    def load_facial_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        face_dict = {k: v for k, v in checkpoint.items() if
                     'face_backbone' in k or k in self.face_backbone.state_dict()}
        self.face_backbone.load_state_dict(face_dict, strict=False)
