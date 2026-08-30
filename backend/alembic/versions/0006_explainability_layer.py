"""Persist model explanations and attention/evidence agreement."""

import sqlalchemy as sa
from alembic import op


revision = "0006_explainability_layer"
down_revision = "0005_evidence_layer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("explainability_results"):
        return
    op.create_table(
        "explainability_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("screening_session_id", sa.Uuid(), nullable=False),
        sa.Column("fundus_image_id", sa.Uuid(), nullable=False),
        sa.Column("predicted_class", sa.Integer(), nullable=False),
        sa.Column("predicted_class_label", sa.String(length=80), nullable=False),
        sa.Column("model_version", sa.String(length=80), nullable=False),
        sa.Column("heatmap_data_uri", sa.Text(), nullable=True),
        sa.Column("overlay_data_uri", sa.Text(), nullable=True),
        sa.Column("normalized_attention_map_data_uri", sa.Text(), nullable=True),
        sa.Column("lesion_evidence_map_data_uri", sa.Text(), nullable=True),
        sa.Column("attention_agreement_status", sa.String(length=40), nullable=False),
        sa.Column("attention_agreement_score", sa.Float(), nullable=True),
        sa.Column("attention_agreement_metrics", sa.JSON(), nullable=True),
        sa.Column("explanation_stability", sa.JSON(), nullable=True),
        sa.Column("counterfactual", sa.JSON(), nullable=True),
        sa.Column("result_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["fundus_image_id"], ["fundus_images.id"]),
        sa.ForeignKeyConstraint(["screening_session_id"], ["screening_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("screening_session_id"),
    )
    op.create_index("ix_explainability_results_screening_session_id", "explainability_results", ["screening_session_id"], unique=True)
    op.create_index("ix_explainability_results_fundus_image_id", "explainability_results", ["fundus_image_id"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("explainability_results"):
        op.drop_table("explainability_results")
