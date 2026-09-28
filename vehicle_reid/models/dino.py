"""Определяет DINOv2 backbone и проекционную голову Vehicle ReID."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


DIMENSIONS = {"dinov2_vits14": 384, "dinov2_vitb14": 768}


def make_backbone(architecture: str, source_dir: str | Path, weights: str | Path | None = None) -> nn.Module:
    """Создает локальный DINOv2 backbone и при необходимости загружает исходные веса."""
    if architecture not in DIMENSIONS:
        raise ValueError(f"unsupported architecture: {architecture}")
    os.environ["XFORMERS_DISABLED"] = "1"
    backbone = torch.hub.load(str(Path(source_dir).resolve()), architecture, source="local", pretrained=False)
    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
    if int(backbone.embed_dim) != DIMENSIONS[architecture] or int(backbone.patch_size) != 14:
        raise ValueError("DINOv2 factory dimensions do not match the selected architecture")
    return backbone


class DinoReID(nn.Module):
    """Модель DINOv2 с проекцией признака и классификатором для обучения."""
    def __init__(self, architecture: str, source_dir: str | Path, initial_weights: str | Path | None,
                 num_classes: int, embedding_dim: int = 384) -> None:
        """Создает backbone, проекцию признака и классификационную голову."""
        super().__init__()
        self.architecture = architecture
        self.backbone = make_backbone(architecture, source_dir, initial_weights)
        self.embedding = nn.Linear(DIMENSIONS[architecture], embedding_dim, bias=False)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.backbone.requires_grad_(False)
        self.trainable_blocks = 2

    def set_epoch(self, epoch: int, warmup_epochs: int = 2) -> None:
        """Размораживает последние блоки backbone после начальных эпох."""
        enabled = epoch >= warmup_epochs
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if enabled:
            for block in self.backbone.blocks[-self.trainable_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)

    def _cls(self, images: torch.Tensor) -> torch.Tensor:
        """Получает CLS-токен с экономией памяти при обучении backbone."""
        if not self.training or not torch.is_grad_enabled():
            return self.backbone.forward_features(images)["x_norm_clstoken"]
        x = self.backbone.prepare_tokens_with_masks(images, masks=None)
        split = len(self.backbone.blocks) - self.trainable_blocks
        for index, block in enumerate(self.backbone.blocks):
            if index < split:
                with torch.no_grad():
                    x = block(x)
            else:
                x = checkpoint(block, x, use_reentrant=False)
        x = self.backbone.norm(x)
        return x[:, 0]

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Возвращает L2-нормированный признак для поиска."""
        raw = self.embedding(self._cls(images))
        return nn.functional.normalize(raw.float(), dim=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает признак и logits классов для обучения."""
        raw = self.embedding(self._cls(images))
        logits = self.classifier(self.bn(raw.float()))
        return nn.functional.normalize(raw.float(), dim=1), logits


def model_state(checkpoint_path: str | Path) -> dict:
    """Читает веса модели из сохраненного checkpoint."""
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint_data["model"] if isinstance(checkpoint_data, dict) and "model" in checkpoint_data else checkpoint_data
