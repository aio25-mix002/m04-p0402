"""Load application configuration from a YAML file.

This configuration manager reads the YAML file located in the
``configs/`` directory and returns an ``AppSettings`` instance composed
of nested dataclasses.  Unlike the original project, we avoid using
``pydantic_settings`` here to minimise external dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Dict

import yaml

from .appsettings import AppSettings, DataSettings, FeatureEngineeringSettings, ModelSettings


class ConfigurationManager:
    """Singleton loader for application settings."""

    _settings: Optional[AppSettings] = None

    @classmethod
    def load(
        cls,
        reload: bool = False,
        config_path: Optional[str | Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> AppSettings:
        """Load the configuration from YAML.

        Parameters
        ----------
        reload: bool, optional
            Force reloading the configuration from disk even if it was
            previously loaded.
        config_path: str or Path, optional
            Path to the YAML configuration file.  If not provided, the
            default location ``<project_root>/configs/config.yaml`` is used.
        overrides: dict, optional
            Key–value pairs to override values loaded from YAML.  Only
            top‑level keys in ``AppSettings`` are supported.

        Returns
        -------
        AppSettings
            Parsed configuration object.
        """
        if cls._settings is None or reload:
            cfg_path = (
                Path(config_path)
                if config_path is not None
                # The project root is two levels above this file (src/configuration -> src -> project root)
                else Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
            )
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_dict: Dict[str, Any] = yaml.safe_load(f) or {}
            # Apply overrides to the root dictionary
            if overrides:
                cfg_dict.update(overrides)
            data_cfg = cfg_dict.get("data", {})
            fe_cfg = cfg_dict.get("feature_engineering", {})
            model_cfg = cfg_dict.get("model", {})
            cls._settings = AppSettings(
                random_state=cfg_dict.get("random_state", 42),
                data=DataSettings(**data_cfg),
                feature_engineering=FeatureEngineeringSettings(**fe_cfg),
                model=ModelSettings(
                    algorithms=model_cfg.get("algorithms", []),
                    params=model_cfg.get("params", {}),
                ),
            )
        return cls._settings
