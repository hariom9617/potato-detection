"""
Application configuration using Pydantic Settings.
Environment variables override defaults.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Configuration
    api_title: str = "Potato Quality Inspection API"
    api_version: str = "1.0.0"
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    
    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://potato_user:potato_dev_password@localhost:5432/potato_inspection_dev",
        env="DATABASE_URL"
    )
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # ML Configuration
    model_path: Optional[str] = Field(default=None, env="MODEL_PATH")
    device: str = Field(default="cpu", env="DEVICE")
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    detection_threshold: float = Field(default=0.25, env="DETECTION_THRESHOLD")
    
    # Storage Configuration
    data_root: str = Field(default="data", env="DATA_ROOT")
    upload_dir: str = Field(default="data/uploads", env="UPLOAD_DIR")
    models_dir: str = Field(default="models", env="MODELS_DIR")
    inspection_images_dir: str = Field(default="data/inspections", env="INSPECTION_IMAGES_DIR")
    
    # File Upload Limits
    max_upload_size_mb: int = Field(default=5, env="MAX_UPLOAD_SIZE_MB")
    min_image_dimension: int = Field(default=224, env="MIN_IMAGE_DIMENSION")
    max_image_dimension: int = Field(default=4096, env="MAX_IMAGE_DIMENSION")
    allowed_image_formats: list[str] = Field(default=["JPEG", "PNG"], env="ALLOWED_IMAGE_FORMATS")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default="logs/app.log", env="LOG_FILE")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Configuration
    secret_key: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Feature Flags
    enable_persistence: bool = Field(default=True, env="ENABLE_PERSISTENCE")
    enable_image_storage: bool = Field(default=True, env="ENABLE_IMAGE_STORAGE")
    
    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_upload_size_mb * 1024 * 1024
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.inspection_images_dir, exist_ok=True)
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()
