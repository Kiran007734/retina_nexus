"""Create the initial RETINA-NEXUS data model.

The metadata-driven migration keeps the first scaffold synchronized with the
SQLAlchemy models while the schema is still evolving.
"""

from alembic import op

from app.database.base import Base
import app.models  # noqa: F401

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
