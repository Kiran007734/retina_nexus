"""Persist retinal clinical-evidence module results."""

import sqlalchemy as sa
from alembic import op

revision = "0005_evidence_layer"
down_revision = "0004_classifier_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("segmentation_results"):
        op.create_table(
            "segmentation_results",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("screening_session_id", sa.Uuid(), nullable=False),
            sa.Column("fundus_image_id", sa.Uuid(), nullable=False),
            sa.Column("structure_type", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("implementation", sa.String(length=120), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("pixel_count", sa.Integer(), nullable=True),
            sa.Column("mask_data_uri", sa.Text(), nullable=True),
            sa.Column("bounding_regions", sa.JSON(), nullable=True),
            sa.Column("result_metadata", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["fundus_image_id"], ["fundus_images.id"]),
            sa.ForeignKeyConstraint(["screening_session_id"], ["screening_sessions.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_segmentation_results_screening_session_id", "segmentation_results", ["screening_session_id"])
        op.create_index("ix_segmentation_results_fundus_image_id", "segmentation_results", ["fundus_image_id"])
        op.create_index("ix_segmentation_results_structure_type", "segmentation_results", ["structure_type"])
    if not inspector.has_table("lesion_results"):
        op.create_table(
            "lesion_results",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("screening_session_id", sa.Uuid(), nullable=False),
            sa.Column("fundus_image_id", sa.Uuid(), nullable=False),
            sa.Column("lesion_type", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("implementation", sa.String(length=120), nullable=False),
            sa.Column("lesion_count", sa.Integer(), nullable=True),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("mask_data_uri", sa.Text(), nullable=True),
            sa.Column("bounding_regions", sa.JSON(), nullable=True),
            sa.Column("result_metadata", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["fundus_image_id"], ["fundus_images.id"]),
            sa.ForeignKeyConstraint(["screening_session_id"], ["screening_sessions.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_lesion_results_screening_session_id", "lesion_results", ["screening_session_id"])
        op.create_index("ix_lesion_results_fundus_image_id", "lesion_results", ["fundus_image_id"])
        op.create_index("ix_lesion_results_lesion_type", "lesion_results", ["lesion_type"])
    if not inspector.has_table("anatomical_landmarks"):
        op.create_table(
            "anatomical_landmarks",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("screening_session_id", sa.Uuid(), nullable=False),
            sa.Column("fundus_image_id", sa.Uuid(), nullable=False),
            sa.Column("landmark_type", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=40), nullable=False),
            sa.Column("method", sa.String(length=120), nullable=False),
            sa.Column("x", sa.Float(), nullable=False),
            sa.Column("y", sa.Float(), nullable=False),
            sa.Column("radius", sa.Float(), nullable=True),
            sa.Column("x_normalized", sa.Float(), nullable=True),
            sa.Column("y_normalized", sa.Float(), nullable=True),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("result_metadata", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["fundus_image_id"], ["fundus_images.id"]),
            sa.ForeignKeyConstraint(["screening_session_id"], ["screening_sessions.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_anatomical_landmarks_screening_session_id", "anatomical_landmarks", ["screening_session_id"])
        op.create_index("ix_anatomical_landmarks_fundus_image_id", "anatomical_landmarks", ["fundus_image_id"])
        op.create_index("ix_anatomical_landmarks_landmark_type", "anatomical_landmarks", ["landmark_type"])


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    for table in ("anatomical_landmarks", "lesion_results", "segmentation_results"):
        if inspector.has_table(table):
            op.drop_table(table)
