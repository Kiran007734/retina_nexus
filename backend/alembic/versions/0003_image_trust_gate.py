"""Store Image Trust Gate results and enhancement artifacts."""

import sqlalchemy as sa
from alembic import op

revision = "0003_image_trust_gate"
down_revision = "0002_dataset_governance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for value in ("GRADABLE", "BORDERLINE", "UNGRADABLE"):
            op.execute(f"ALTER TYPE qualitydecision ADD VALUE IF NOT EXISTS '{value}'")
    existing_columns = {column["name"] for column in sa.inspect(bind).get_columns("fundus_images")}
    if "quality_assessment" not in existing_columns:
        op.add_column("fundus_images", sa.Column("quality_assessment", sa.JSON(), nullable=True))
    if "quality_checked_at" not in existing_columns:
        op.add_column("fundus_images", sa.Column("quality_checked_at", sa.DateTime(timezone=True), nullable=True))
    if "enhanced_storage_path" not in existing_columns:
        op.add_column("fundus_images", sa.Column("enhanced_storage_path", sa.String(length=512), nullable=True))
    if "enhancement_passes" not in existing_columns:
        op.add_column("fundus_images", sa.Column("enhancement_passes", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("fundus_images", "enhancement_passes")
    op.drop_column("fundus_images", "enhanced_storage_path")
    op.drop_column("fundus_images", "quality_checked_at")
    op.drop_column("fundus_images", "quality_assessment")
