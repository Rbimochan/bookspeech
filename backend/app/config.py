from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    storage_dir: Path = Path("../storage")
    max_concurrent_jobs: int = 1
    default_voice: str = "af_heart"
    default_speed: float = 0.9
    db_path: Path = Path("../storage/bookspeech.db")

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.storage_dir / "output"

    @property
    def logs_dir(self) -> Path:
        return self.storage_dir / "logs"


settings = Settings()

for d in (settings.uploads_dir, settings.output_dir, settings.logs_dir):
    d.mkdir(parents=True, exist_ok=True)
