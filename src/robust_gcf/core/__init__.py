"""Cross-environment contracts. Imports only the stdlib and pydantic (no torch / tf / lightning)."""
from robust_gcf.core.schemas import (  # noqa: F401
    CacheManifest,
    CFRecord,
    ClassifierMeta,
    EvalItem,
    PreprocessSpec,
    TargetSpec,
)
