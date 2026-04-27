import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    dir: Path
    img_size: int = Field(224, ge=32, le=1024)
    batch_size: int = Field(32, ge=1)
    num_workers: int = Field(4, ge=0)
    val_split: float = Field(0.15, ge=0.0, le=1.0)
    test_split: float = Field(0.15, ge=0.0, le=1.0)
    seed: int = Field(42, ge=0)

    @field_validator("dir", mode="before")
    @classmethod
    def validate_dir(cls, v: Path) -> Path:
        return Path(v)


class ModelConfig(BaseModel):
    num_classes: int = Field(5, ge=1)
    backbone: str = Field("convnext_base")
    transformer_dim: int = Field(512, ge=64)
    transformer_heads: int = Field(8, ge=1)
    transformer_layers: int = Field(4, ge=1)
    mlp_ratio: float = Field(4.0, ge=1.0)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    drop_path_rate: float = Field(0.1, ge=0.0, le=1.0)
    pretrained: bool = True


class TrainingConfig(BaseModel):
    learning_rate: float = Field(1e-4, gt=0.0)
    backbone_lr_scale: float = Field(0.1, gt=0.0)
    weight_decay: float = Field(1e-5, ge=0.0)
    epochs: int = Field(100, ge=1)
    warmup_epochs: int = Field(5, ge=0)
    label_smoothing: float = Field(0.1, ge=0.0, le=1.0)
    mixup_alpha: float = Field(0.2, ge=0.0)
    grad_clip: float = Field(1.0, ge=0.0)
    early_stopping_patience: int = Field(10, ge=0)


class OutputConfig(BaseModel):
    dir: Path = Field(Path("outputs"))

    @field_validator("dir", mode="before")
    @classmethod
    def validate_dir(cls, v: Path) -> Path:
        return Path(v)


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig

    @classmethod
    def from_toml(cls, path: Path) -> "Config":

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls(**data)
