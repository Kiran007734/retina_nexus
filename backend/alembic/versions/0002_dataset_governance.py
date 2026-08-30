"""Add dataset governance tables."""

from alembic import op

from app.database.base import Base
import app.models  # noqa: F401

revision = "0002_dataset_governance"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    # Dataset tables are additive in this scaffold; controlled deployments should
    # provide an explicit data-retention decision before dropping them.
    pass
