"""Path configuration for OMAD.

Parameter defaults (theta_g, seeds, ratios, etc.) live in config.yaml
and are passed through the CLI. This module only defines path conventions.
"""

from pathlib import Path
from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """Path configuration for OMAD."""

    # Use current working directory as base
    project_root: Path = Path(".")
    data_dir: Path = Path("./data")

    def route_sliced_dir(self, T: int) -> Path:
        """Get route_sliced_{T} directory path."""
        return self.data_dir / f"route_sliced_{T}"

    def preprocessed_csv(self, T: int) -> Path:
        """Get preprocessed CSV path for given T."""
        return self.route_sliced_dir(T) / f"routes_sliced_{T}_preprocessed.csv"

    def injected_csv(self, T: int) -> Path:
        """Get injected CSV path for given T."""
        return self.route_sliced_dir(T) / f"routes_sliced_{T}_injected.csv"

    def raw_routes_csv(self, T: int) -> Path:
        """Get raw routes CSV path for given T."""
        return self.route_sliced_dir(T) / f"routes_sliced_{T}.csv"

    def user_query_dir(self, T: int) -> Path:
        """Get user_query directory path for given T."""
        return self.route_sliced_dir(T) / "user_query"

    def indices_dir(self, T: int) -> Path:
        """Get indices directory path for given T."""
        return self.route_sliced_dir(T) / "indices"

    def injected_json_dir(self, T: int) -> Path:
        """Get injected JSON directory path for given T."""
        return self.route_sliced_dir(T) / "injected"


class OmadConfig(BaseModel):
    """Master configuration for OMAD (paths only)."""

    paths: PathConfig = Field(default_factory=PathConfig)


# Global config instance
_config: OmadConfig | None = None


def get_config() -> OmadConfig:
    """Get global config instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = OmadConfig()
    return _config
