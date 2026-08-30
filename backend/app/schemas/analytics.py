from pydantic import BaseModel, ConfigDict, Field


class AnalyticsOverviewResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    total_screenings: int = 0
    today_screenings: int = 0
    referable_cases: int = 0
    human_review_cases: int = 0
    ungradable_images: int = 0
    completed_screenings: int = 0
    status_distribution: dict[str, int] = Field(default_factory=dict)
    severity_distribution: dict[str, int] = Field(default_factory=dict)
    recent_activity: list[dict] = Field(default_factory=list)
    system_health: dict = Field(default_factory=dict)
