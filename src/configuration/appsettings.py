"""Plain dataclass representations of the configuration schema.

This module defines simple data classes that mirror the expected
structure of ``configs/config.yaml``.  We avoid heavy dependencies
like ``pydantic`` and instead perform light validation in the
``ConfigurationManager``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DataSettings:
    raw_file: str
    interim_dir: str
    processed_dir: str
    test_size: float
    val_size: float


@dataclass
class FeatureEngineeringSettings:
    add_ratio_features: bool = False


@dataclass
class ModelSettings:
    algorithms: List[str] = field(default_factory=list)
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class AppSettings:
    random_state: int
    data: DataSettings
    feature_engineering: FeatureEngineeringSettings
    model: ModelSettings
