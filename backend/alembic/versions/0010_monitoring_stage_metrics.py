"""Store per-stage timing for operational monitoring."""

import sqlalchemy as sa
from alembic import op


revision = "0010_monitoring_stage_metrics"
down_revision = "0009_human_review_and_reports"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("screening_runs"):
        columns = {column["name"] for column in inspector.get_columns("screening_runs")}
        if "stage_metrics" not in columns:
            op.add_column("screening_runs", sa.Column("stage_metrics", sa.JSON(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("screening_runs"):
        columns = {column["name"] for column in inspector.get_columns("screening_runs")}
        if "stage_metrics" in columns:
            op.drop_column("screening_runs", "stage_metrics")
