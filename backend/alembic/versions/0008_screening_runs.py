"""Persist end-to-end screening orchestration state and artifact."""

import sqlalchemy as sa
from alembic import op


revision = "0008_screening_runs"
down_revision = "0007_retinaguard"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("screening_runs"):
        return
    op.create_table(
        "screening_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("fundus_image_id", sa.Uuid(), nullable=False),
        sa.Column("initiating_user_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("stage_status", sa.JSON(), nullable=False),
        sa.Column("stage_errors", sa.JSON(), nullable=True),
        sa.Column("quality", sa.JSON(), nullable=True),
        sa.Column("classification", sa.JSON(), nullable=True),
        sa.Column("lesions", sa.JSON(), nullable=True),
        sa.Column("explainability", sa.JSON(), nullable=True),
        sa.Column("retinaguard", sa.JSON(), nullable=True),
        sa.Column("triage", sa.JSON(), nullable=True),
        sa.Column("model_versions", sa.JSON(), nullable=True),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["fundus_image_id"], ["fundus_images.id"]),
        sa.ForeignKeyConstraint(["id"], ["screening_sessions.id"]),
        sa.ForeignKeyConstraint(["initiating_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_screening_runs_fundus_image_id", "screening_runs", ["fundus_image_id"])
    op.create_index("ix_screening_runs_initiating_user_id", "screening_runs", ["initiating_user_id"])
    op.create_index("ix_screening_runs_status", "screening_runs", ["status"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("screening_runs"):
        op.drop_table("screening_runs")
