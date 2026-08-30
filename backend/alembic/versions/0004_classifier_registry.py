"""Store classifier training provenance in the model registry."""

import sqlalchemy as sa
from alembic import op

revision = "0004_classifier_registry"
down_revision = "0003_image_trust_gate"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("model_versions")}
    if "training_config" not in columns:
        op.add_column("model_versions", sa.Column("training_config", sa.JSON(), nullable=True))
    if "dataset_version" not in columns:
        op.add_column("model_versions", sa.Column("dataset_version", sa.String(length=120), nullable=True))


def downgrade() -> None:
    op.drop_column("model_versions", "dataset_version")
    op.drop_column("model_versions", "training_config")
