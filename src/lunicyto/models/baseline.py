import timm
import torch
import torch.nn as nn


class ConvNextBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "convnext_base",
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        # global_pool="avg" возвращает вектор (B, feat_dim)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim: int = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.head[-1].bias)
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # (B, feat_dim)
        return self.head(feats)


def build_baseline_model(
    num_classes: int = 5,
    backbone: str = "convnext_base",
    dropout: float = 0.3,
    pretrained: bool = True,
) -> ConvNextBaseline:
    return ConvNextBaseline(
        num_classes=num_classes,
        backbone=backbone,
        dropout=dropout,
        pretrained=pretrained,
    )
