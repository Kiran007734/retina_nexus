"""Persist transparent RetinaGuard self-check results."""

import sqlalchemy as sa
from alembic import op


revision = "0007_retinaguard"
down_revision = "0006_explainability_layer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("retinaguard_results"):
        return
    op.create_table(
        "retinaguard_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("screening_session_id", sa.Uuid(), nullable=False),
        sa.Column("fundus_image_id", sa.Uuid(), nullable=False),
        sa.Column("trust_score", sa.Float(), nullable=False),
        sa.Column("trust_category", sa.String(length=40), nullable=False),
        sa.Column("contributing_factors", sa.JSON(), nullable=False),
        sa.Column("risk_flags", sa.JSON(), nullable=False),
        sa.Column("recommended_action", sa.String(length=512), nullable=False),
        sa.Column("calibration", sa.JSON(), nullable=True),
        sa.Column("uncertainty", sa.JSON(), nullable=True),
        sa.Column("model_disagreement", sa.JSON(), nullable=True),
        sa.Column("ood", sa.JSON(), nullable=True),
        sa.Column("signal_snapshot", sa.JSON(), nullable=True),
        sa.Column("configuration", sa.JSON(), nullable=True),
        sa.Column("reason_summary", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["fundus_image_id"], ["fundus_images.id"]),
        sa.ForeignKeyConstraint(["screening_session_id"], ["screening_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("screening_session_id"),
    )
    op.create_index("ix_retinaguard_results_screening_session_id", "retinaguard_results", ["screening_session_id"], unique=True)
    op.create_index("ix_retinaguard_results_fundus_image_id", "retinaguard_results", ["fundus_image_id"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("retinaguard_results"):
        op.drop_table("retinaguard_results")
