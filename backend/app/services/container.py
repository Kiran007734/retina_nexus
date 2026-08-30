from functools import lru_cache

from app.services.placeholders import (
    PlaceholderDRClassificationService,
    PlaceholderEvidenceVerificationService,
    PlaceholderExplainabilityService,
    PlaceholderImagePreprocessingService,
    PlaceholderImageQualityService,
    PlaceholderLesionDetectionService,
    PlaceholderReportGenerationService,
)
from app.services.orchestrator import ScreeningOrchestrator
from app.ml.quality.trust_gate import ImageTrustGateService
from app.ml.inference.classifier import TorchDRClassificationService
from app.ml.models.classifier import ReferableDRMapping
from app.ml.evidence.service import RetinalEvidenceService
from app.ml.evidence.lesion_model import MODEL_CLASS_TO_MODULE, PretrainedRetinalLesionAdapter
from app.ml.evidence.vessel_model import PretrainedRetinalVesselAdapter
from app.ml.explainability.service import ExplainabilityService
from app.ml.trust.calibration import TemperatureScaler
from app.ml.trust.guard import RetinaGuardEngine
from app.ml.trust.ood import FeatureDistributionMonitor
from app.ml.trust.uncertainty import UncertaintyEstimator
from app.services.screening_pipeline import ScreeningPipelineService
from app.storage.container import get_storage
from app.core.config import get_settings


@lru_cache
def get_service_container() -> dict:
    """Single composition point for swapping implementations without changing routes."""
    services = {
        "quality": ImageTrustGateService(),
        "preprocessing": PlaceholderImagePreprocessingService(),
        "classification": PlaceholderDRClassificationService(),
        "lesions": PlaceholderLesionDetectionService(),
        "explainability": PlaceholderExplainabilityService(),
        "evidence": PlaceholderEvidenceVerificationService(),
        "trust": get_retinaguard_service(),
        "reports": PlaceholderReportGenerationService(),
    }
    services["orchestrator"] = ScreeningOrchestrator(services)
    return services


@lru_cache
def get_image_quality_service() -> ImageTrustGateService:
    """Dependency-injection entry point for the first vision stage."""
    return ImageTrustGateService()


@lru_cache
def get_classifier_service() -> TorchDRClassificationService:
    settings = get_settings()
    return TorchDRClassificationService(
        model_path=settings.classifier_model_path,
        backbone=settings.classifier_backbone,
        model_version=settings.classifier_model_version,
        device=settings.classifier_device,
        referable_mapping=ReferableDRMapping(
            name=f"grade_{settings.referable_min_grade}_or_worse",
            referable_grades=tuple(range(settings.referable_min_grade, 5)),
        ),
    )


@lru_cache
def get_evidence_service() -> RetinalEvidenceService:
    settings = get_settings()
    lesion_adapter = PretrainedRetinalLesionAdapter(
        model_path=settings.lesion_model_path,
        device=settings.lesion_model_device,
        threshold=settings.lesion_model_threshold,
        version=settings.lesion_model_version,
    )
    vessel_adapter = PretrainedRetinalVesselAdapter(
        model_path=settings.vessel_model_path,
        device=settings.vessel_model_device,
        threshold=settings.vessel_model_threshold,
        version=settings.vessel_model_version,
    )
    adapters = {"vessel_segmentation": vessel_adapter}
    if lesion_adapter.is_configured:
        adapters.update({module: lesion_adapter for module in MODEL_CLASS_TO_MODULE.values()})
    return RetinalEvidenceService(
        max_dimension=settings.evidence_max_dimension,
        enable_heuristics=settings.evidence_enable_heuristics,
        model_adapters=adapters,
        enable_vessel_baseline=settings.evidence_enable_vessel_baseline,
    )


@lru_cache
def get_explainability_service() -> ExplainabilityService:
    settings = get_settings()
    return ExplainabilityService(
        classifier=get_classifier_service(),
        stability_enabled=settings.explainability_stability_enabled,
        counterfactual_enabled=settings.explainability_counterfactual_enabled,
        max_stability_variants=settings.explainability_max_stability_variants,
    )


@lru_cache
def get_retinaguard_service() -> RetinaGuardEngine:
    settings = get_settings()
    return RetinaGuardEngine(
        version=settings.retinaguard_config_version,
        calibrator=TemperatureScaler(
            temperature=settings.retinaguard_temperature,
            version=settings.retinaguard_calibration_version,
            fitted=settings.retinaguard_calibration_fitted,
        ),
        uncertainty_estimator=UncertaintyEstimator(),
        ood_monitor=FeatureDistributionMonitor(settings.retinaguard_ood_reference_path, settings.retinaguard_ood_threshold),
        mc_dropout_enabled=settings.retinaguard_mc_dropout_enabled,
        mc_dropout_samples=settings.retinaguard_mc_dropout_samples,
        weights={
            "quality": settings.retinaguard_weight_quality,
            "calibrated_confidence": settings.retinaguard_weight_calibrated_confidence,
            "uncertainty": settings.retinaguard_weight_uncertainty,
            "model_agreement": settings.retinaguard_weight_model_agreement,
            "lesion_evidence": settings.retinaguard_weight_lesion_evidence,
            "attention_lesion_agreement": settings.retinaguard_weight_attention_lesion_agreement,
            "explanation_stability": settings.retinaguard_weight_explanation_stability,
            "ood": settings.retinaguard_weight_ood,
        },
        missing_signal_score=settings.retinaguard_missing_signal_score,
        trusted_threshold=settings.retinaguard_trusted_threshold,
        unreliable_threshold=settings.retinaguard_unreliable_threshold,
    )


@lru_cache
def get_screening_pipeline_service() -> ScreeningPipelineService:
    return ScreeningPipelineService(
        quality_service=get_image_quality_service(),
        classifier=get_classifier_service(),
        evidence_service=get_evidence_service(),
        explainability_service=get_explainability_service(),
        retinaguard=get_retinaguard_service(),
        storage=get_storage(),
    )
