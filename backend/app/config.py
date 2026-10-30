from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WIND_", env_file=".env")

    data_dir: Path = ROOT / "data" / "processed"
        default_cutoff: str = "09:20"  # UK local time on D-1: EPEX GB day-ahead gate closure
    elexon_base_url: str = "https://data.elexon.co.uk/bmrs/api/v1"  # verify in Phase 2


settings = Settings()