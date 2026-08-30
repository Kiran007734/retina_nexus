"""Track model artifact kind, lifecycle, availability, and load errors."""

import sqlalchemy as sa
from alembic import op


revision = "0011_model_registry_artifact_status"
down_revision = "0010_monitoring_stage_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("model_versions"):
        return
    columns = {column["name"] for column in inspector.get_columns("model_versions")}
    if "artifact_kind" not in columns:
        op.add_column("model_versions", sa.Column("artifact_kind", sa.String(length=48), nullable=False, server_default="FINE_TUNED_MODEL"))
    if "artifact_status" not in columns:
        op.add_column("model_versions", sa.Column("artifact_status", sa.String(length=40), nullable=False, server_default="MODEL_MISSING"))
    if "availability_status" not in columns:
        op.add_column("model_versions", sa.Column("availability_status", sa.String(length=40), nullable=False, server_default="MODEL_MISSING"))
    if "load_error" not in columns:
        op.add_column("model_versions", sa.Column("load_error", sa.String(length=1000), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("model_versions"):
        return
    columns = {column["name"] for column in inspector.get_columns("model_versions")}
    for name in ("load_error", "availability_status", "artifact_status", "artifact_kind"):
        if name in columns:
            op.drop_column("model_versions", name)
