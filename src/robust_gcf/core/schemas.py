"""Schemas shared by every environment (training env and each CF-method env).

Constraints: stdlib + pydantic only, works with pydantic v1 and v2, Python >= 3.8.
(No `X | None` annotations and no validators, whose APIs differ between pydantic versions.)

Conventions that hold everywhere:
  * Images exchanged between components are float32 in [0, 1], NCHW, square.
  * Binary labels: 1 = positive = CF *source* class (Smiling, Young, Effusion),
    0 = negative = CF *target* class. All CFs go 1 -> 0.
  * Classifiers output ONE logit for the positive class; p = sigmoid(logit).
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

try:  # Literal is in typing from 3.8
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore

PYDANTIC_V2 = hasattr(BaseModel, "model_dump")

DatasetName = Literal["celeba", "chestmnist"]
SplitName = Literal["train", "val", "test"]
PathLike = Union[str, Path]

if PYDANTIC_V2:
    from pydantic import ConfigDict

    class _Schema(BaseModel):
        model_config = ConfigDict(extra="forbid", protected_namespaces=())

else:  # pragma: no cover - pydantic v1

    class _Schema(BaseModel):  # type: ignore[no-redef]
        class Config:
            extra = "forbid"


def _dump(m: BaseModel) -> Dict[str, Any]:
    if PYDANTIC_V2:
        return m.model_dump(mode="json")
    return json.loads(m.json())  # pragma: no cover


def _validate(cls, data: Dict[str, Any]):
    if PYDANTIC_V2:
        return cls.model_validate(data)
    return cls.parse_obj(data)  # pragma: no cover


class Schema(_Schema):
    """Version-agnostic (de)serialization helpers."""

    def to_dict(self) -> Dict[str, Any]:
        return _dump(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return _validate(cls, dict(data))

    def replace(self, **changes):
        return self.from_dict({**self.to_dict(), **changes})

    def save(self, path: PathLike) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: PathLike):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- data


class TargetSpec(Schema):
    """Maps a dataset attribute/finding to a binary task."""

    name: str  # e.g. "celeba_smiling"; also used in output paths
    dataset: DatasetName
    column: str  # label column in the cache index ("Smiling", "Young", "effusion")
    positive_name: str  # CF source class, label 1
    negative_name: str  # CF target class, label 0
    # "absent": negatives are all rows with column == 0.
    # "no_finding": negatives are rows where *every* label column is 0 (ChestMNIST "Healthy").
    negative_policy: Literal["absent", "no_finding"] = "absent"
    # How class imbalance is handled in training; fixed per task and recorded.
    imbalance: Literal["none", "pos_weight", "threshold"] = "none"


class CacheManifest(Schema):
    """Written last when a cache build finishes; its presence marks a complete cache."""

    dataset: DatasetName
    resolution: int
    channels: int
    crop: str
    num_images: int
    splits: Dict[str, int]
    label_columns: List[str]
    source: str
    format_version: int = 1


# ---------------------------------------------------------------------- classifier


class PreprocessSpec(Schema):
    """What a classifier does to a [0,1] image before the backbone."""

    input_size: int = 128
    channels: Literal[1, 3] = 3
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    interpolation: Literal["bilinear"] = "bilinear"
    antialias: bool = True

    @classmethod
    def default_for(cls, channels: int, input_size: int = 128) -> "PreprocessSpec":
        if channels == 3:
            return cls(input_size=input_size, channels=3)
        if channels == 1:
            return cls(input_size=input_size, channels=1, mean=(0.5,), std=(0.5,))
        raise ValueError(f"unsupported channel count: {channels}")


class ClassifierMeta(Schema):
    """Stored next to every classifier checkpoint (meta.json)."""

    arch: str = "densenet121"
    target: TargetSpec
    preprocess: PreprocessSpec
    threshold: float = 0.5
    threshold_method: str = "fixed"
    variant: str = "m0"  # m0, seed_<k>, subsample_<k>, weight_noise_<sigma>, ...
    variant_params: Dict[str, Any] = {}
    seed: int = 0
    train_subsample: Optional[float] = None
    pos_weight: Optional[float] = None
    cache_resolution: int = 128
    metrics: Dict[str, float] = {}
    lib_versions: Dict[str, str] = {}
    format_version: int = 1


# ------------------------------------------------------------- evaluation / CF files


class EvalItem(Schema):
    """One row of outputs/evalsets/{dataset}/{target}/manifest.parquet."""

    eval_id: int
    cache_index: int
    image_id: str
    label: int
    p_m0: float
    perturbation_id: int = 0  # 0 = unperturbed factual
    perturbation: Optional[str] = None  # e.g. "gaussian"
    perturbation_param: Optional[float] = None
    perturbation_seed: Optional[int] = None


class CFRecord(Schema):
    """One row of outputs/cfs/{method}/{dataset}/{target}/meta.parquet (draft)."""

    method: str
    dataset: DatasetName
    target: str
    eval_id: int
    perturbation_id: int = 0
    classifier_variant: str = "m0"
    run_seed: int = 0  # generation seed (DiME needs >= 2)
    success: bool
    p_factual: float
    p_cf: float
    runtime_s: Optional[float] = None
    extras: Dict[str, Any] = {}
