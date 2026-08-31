from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=("settings_",))

    app_name: str = "RETINA-NEXUS"
    environment: str = "development"
    log_level: str = "INFO"
    secret_key: str = ""
    access_token_expire_minutes: int = 60
    cors_origins: list[str] = Field(default=["http://localhost:5173", "http://127.0.0.1:5173"])
    backend_port: int = 8000
    frontend_url: str = "http://localhost:5173"
    model_directory: str = "./ml/weights"
    data_directory: str = "./ml/datasets"
    upload_directory: str = "./storage"
    database_url: str = "sqlite+aiosqlite:///./retina_nexus.db"
    redis_url: str = "redis://localhost:6379/0"
    storage_backend: str = "local"
    local_storage_path: str | None = None
    max_upload_size_mb: int = 15
    allowed_image_mime_types: list[str] = Field(default=["image/jpeg", "image/png"])
    demo_mode_enabled: bool = False
    classifier_model_path: str | None = None
    classifier_backbone: str = "efficientnet_b0"
    classifier_model_version: str | None = None
    classifier_device: str = "auto"
    referable_min_grade: int = Field(default=2, ge=1, le=4)
    evidence_enable_heuristics: bool = True
    evidence_enable_vessel_baseline: bool = False
    evidence_max_dimension: int = Field(default=768, ge=256, le=2048)
    lesion_model_path: str | None = None
    lesion_model_version: str = "fundus-lesions-unet-seresnext50-all-v1"
    lesion_model_device: str = "auto"
    lesion_model_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    vessel_model_path: str | None = None
    vessel_model_version: str = "r2-v2-bv-2025"
    vessel_model_device: str = "auto"
    vessel_model_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    explainability_stability_enabled: bool = False
    explainability_counterfactual_enabled: bool = False
    explainability_max_stability_variants: int = Field(default=3, ge=1, le=5)
    retinaguard_config_version: str = "retinaguard-v2-reliability"
    retinaguard_temperature: float = Field(default=1.0, gt=0)
    retinaguard_calibration_version: str = "temperature-scaling-unfitted"
    retinaguard_calibration_fitted: bool = False
    retinaguard_ood_reference_path: str | None = None
    retinaguard_ood_threshold: float = Field(default=3.0, gt=0)
    retinaguard_missing_signal_score: float = Field(default=0.25, ge=0, le=1)
    retinaguard_trusted_threshold: float = Field(default=0.75, ge=0, le=1)
    retinaguard_unreliable_threshold: float = Field(default=0.45, ge=0, le=1)
    retinaguard_weight_quality: float = Field(default=0.20, ge=0)
    retinaguard_weight_calibrated_confidence: float = Field(default=0.20, ge=0)
    retinaguard_weight_uncertainty: float = Field(default=0.15, ge=0)
    retinaguard_weight_model_agreement: float = Field(default=0.10, ge=0)
    retinaguard_weight_lesion_evidence: float = Field(default=0.10, ge=0)
    retinaguard_weight_attention_lesion_agreement: float = Field(default=0.15, ge=0)
    retinaguard_weight_explanation_stability: float = Field(default=0.05, ge=0)
    retinaguard_weight_ood: float = Field(default=0.05, ge=0)
    retinaguard_mc_dropout_enabled: bool = False
    retinaguard_mc_dropout_samples: int = Field(default=8, ge=2, le=30)

    @model_validator(mode="after")
    def validate_runtime_security(self) -> "Settings":
        if self.environment.lower() in {"production", "prod"} and (len(self.secret_key) < 32 or self.secret_key.startswith("replace-with")):
            raise ValueError("SECRET_KEY must be a non-placeholder value of at least 32 characters in production")
        if self.frontend_url and self.frontend_url not in self.cors_origins:
            self.cors_origins = [*self.cors_origins, self.frontend_url]
        if not self.local_storage_path:
            self.local_storage_path = self.upload_directory
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
