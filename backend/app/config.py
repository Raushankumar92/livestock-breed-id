from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    model_path: Path = Path("models/breed_classifier.keras")
    labels_path: Path = Path("models/breed_labels.json")
    yolo_model: str = "yolov8n.pt"
    enable_detection: bool = True
    max_files: int = 10
    max_file_size_mb: int = 10
    allowed_origins: str = "http://localhost:5173"

    @property
    def cors_origins(self) -> list[str]:
        return [value.strip() for value in self.allowed_origins.split(",") if value.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
