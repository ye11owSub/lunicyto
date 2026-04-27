import timm
import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention с residual + stochastic depth
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.drop_path(attn_out)
        # FFN с residual + stochastic depth
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HybridViTCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "convnext_base",
        transformer_dim: int = 768,
        transformer_heads: int = 12,
        transformer_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.3,
        drop_path_rate: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            cnn_feat_dim = feats.shape[1]
            seq_len = feats.shape[2] * feats.shape[3]

        self.cnn_feat_dim = cnn_feat_dim
        self.seq_len = seq_len

        self.proj = nn.Sequential(
            nn.Linear(cnn_feat_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, transformer_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, transformer_layers)]
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=transformer_dim,
                    num_heads=transformer_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(transformer_layers)
            ]
        )
        self.norm = nn.LayerNorm(transformer_dim)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.head[-1].bias)
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN: (B, 3, H, W) → (B, C, h, w)
        cnn_feats = self.backbone(x)
        B = cnn_feats.shape[0]

        cnn_feats = cnn_feats.flatten(2).transpose(1, 2)

        x_seq = self.proj(cnn_feats)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_tokens, x_seq], dim=1)

        x_seq = x_seq + self.pos_embed

        for blk in self.transformer_blocks:
            x_seq = blk(x_seq)
        x_seq = self.norm(x_seq)

        cls_out = x_seq[:, 0]
        return self.head(cls_out)

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        cnn_feats = self.backbone(x)
        B = cnn_feats.shape[0]
        cnn_feats = cnn_feats.flatten(2).transpose(1, 2)
        x_seq = self.proj(cnn_feats)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_tokens, x_seq], dim=1)
        x_seq = x_seq + self.pos_embed

        attn_maps = []
        for blk in self.transformer_blocks:
            normed = blk.norm1(x_seq)
            _, weights = blk.attn(normed, normed, normed, need_weights=True)
            attn_maps.append(weights.detach())
            x_seq = blk(x_seq)

        return attn_maps


def build_model(
    num_classes: int = 5,
    backbone: str = "convnext_base",
    transformer_dim: int = 768,
    transformer_heads: int = 12,
    transformer_layers: int = 4,
    mlp_ratio: float = 4.0,
    dropout: float = 0.3,
    drop_path_rate: float = 0.1,
    pretrained: bool = True,
) -> HybridViTCNN:
    return HybridViTCNN(
        num_classes=num_classes,
        backbone=backbone,
        transformer_dim=transformer_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        pretrained=pretrained,
    )
