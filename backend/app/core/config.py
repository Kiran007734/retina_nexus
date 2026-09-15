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
    max_image_pixels: int = Field(default=50_000_000, ge=1_000_000, le=100_000_000)
    allowed_image_mime_types: list[str] = Field(default=["image/jpeg", "image/png"])
    demo_mode_enabled: bool = False
    classifier_model_path: str | None = None
    classifier_backbone: str = "efficientnet_b0"
    classifier_model_version: str | None = None
    classifier_device: str = "auto"
    classifier_model_sha256: str | None = None
    # Research-gated referable fusion. Disabled by default because the
    # verifier has known IDRiD training-overlap risk and is not promoted.
    referable_fusion_enabled: bool = False
    referable_fusion_verifier_model_path: str | None = None
    referable_fusion_verifier_model_version: str = "retguard-dr-v1.0.0"
    referable_fusion_verifier_model_sha256: str | None = "f0e19fa86d5a27a05731550d1d6708c01f6f363f45a1fa57849de988f91e775b"
    referable_fusion_threshold: float = Field(default=0.40, ge=0.05, le=0.95)
    lesion_model_sha256: str | None = None
    vessel_model_sha256: str | None = None
    verify_models_on_startup: bool = True
    max_concurrent_screenings: int = Field(default=1, ge=1, le=8)
    screening_timeout_seconds: int = Field(default=900, ge=30, le=3600)
    # The primary path is measured locally at <5 seconds on warm CPU runs;
    # this budget leaves room for cold-start and image-quality variation.
    screening_primary_timeout_seconds: int = Field(default=60, ge=10, le=600)
    # Optional evidence has measured warm CPU runs from ~1.4s to ~310s. The
    # larger bounded budget prevents a genuine CPU inference from being
    # mislabeled as unavailable while keeping enrichment non-blocking for the
    # primary result. It does not alter model inputs or outputs.
    screening_optional_evidence_timeout_seconds: int = Field(default=600, ge=30, le=1800)
    # Grad-CAM/agreement measured ~0.9s-8.4s on completed local runs; this
    # budget is a documented engineering limit, not a clinical target.
    screening_optional_explainability_timeout_seconds: int = Field(default=30, ge=10, le=900)
    referable_min_grade: int = Field(default=2, ge=1, le=4)
    evidence_enable_heuristics: bool = True
    evidence_enable_vessel_baseline: bool = False
    evidence_max_dimension: int = Field(default=768, ge=256, le=2048)
    lesion_model_path: str | None = None
    lesion_model_version: str = "fundus-lesions-unet-seresnext50-all-v1"
    lesion_model_device: str = "auto"
    lesion_model_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    # IDRiD lesion model is research-only and opt-in.  The preserved external
    # lesion model above remains the default evidence adapter.
    idrid_lesion_model_enabled: bool = False
    idrid_lesion_model_path: str | None = None
    idrid_lesion_model_version: str = "idrid-lesion-unet-seresnext50-768-focaldice-20260913-v2"
    idrid_lesion_model_device: str = "auto"
    idrid_lesion_model_threshold: float = Field(default=0.7, ge=0.05, le=0.95)
    idrid_lesion_model_sha256: str | None = None
    # Frozen IDRiD optic-disc/fovea localization is supporting evidence only
    # and remains opt-in so the existing production path is preserved.
    idrid_localization_model_enabled: bool = False
    idrid_localization_model_path: str | None = None
    idrid_localization_model_version: str = "idrid-localization-frozen"
    idrid_localization_model_device: str = "auto"
    idrid_localization_model_sha256: str | None = None
    vessel_model_path: str | None = None
    vessel_model_version: str = "r2-v2-bv-2025"
    vessel_model_device: str = "auto"
    vessel_model_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    # Experimental DRIVE-trained vessel model. Disabled by default so the
    # protected R2-V2 production adapter remains unchanged.
    drive_vessel_model_enabled: bool = False
    drive_vessel_model_path: str | None = None
    drive_vessel_model_version: str = "drive-vessel-scratch-green-focal-dice-512-20260913-v1"
    drive_vessel_model_device: str = "auto"
    drive_vessel_model_threshold: float = Field(default=0.3, ge=0.05, le=0.95)
    drive_vessel_model_sha256: str | None = None
    explainability_stability_enabled: bool = False
    explainability_counterfactual_enabled: bool = False
    explainability_max_stability_variants: int = Field(default=3, ge=1, le=5)
    retinaguard_config_version: str = "retinaguard-v3-graceful-degradation"
    retinaguard_temperature: float = Field(default=1.0, gt=0)
    retinaguard_calibration_version: str = "temperature-scaling-unfitted"
    retinaguard_calibration_fitted: bool = False
    retinaguard_ood_reference_path: str | None = None
    retinaguard_ood_threshold: float = Field(default=3.0, gt=0)
    # Missing signals are excluded and the remaining configured weights are
    # renormalized. There is deliberately no fallback score for unavailable
    # evidence, so configuration cannot reintroduce an invented 25% signal.
    retinaguard_missing_signal_score: float | None = None
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
