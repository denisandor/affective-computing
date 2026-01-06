import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F


class FacialMLP(nn.Module):

    def __init__(
            self,
            num_classes: int,
            input_dim: int = 7,
            hidden_dim: int = 128,
            embedding_dim: int = 64,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_features: bool = False):
        embedding = self.net(x)
        logits = self.classifier(embedding)
        if return_features:
            return logits, embedding
        return logits


class FacialEmotionNet(nn.Module):

    def __init__(
            self,
            num_classes: int,
            embedding_dim: int = 128,
            pretrained: bool = True,
            freeze_backbone: bool = False,
    ):
        super().__init__()

        resnet = models.resnet18(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 512

        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

        if not freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor, return_features: bool = False):

        features = self.backbone(x)

        attn_weights = self.attention(features)
        attended = features * attn_weights

        pooled = self.global_pool(attended).flatten(1)

        embedding = self.embedding(pooled)

        logits = self.classifier(embedding)

        if return_features:
            return logits, embedding
        return logits

    def get_embedding_dim(self):
        return self.embedding_dim
