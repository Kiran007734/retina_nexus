"""Store organization and professional title for self-registered users."""

import sqlalchemy as sa
from alembic import op


revision = "0012_user_profile_fields"
down_revision = "0011_model_registry_artifact_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("users"):
        return
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "organization" not in columns:
        op.add_column("users", sa.Column("organization", sa.String(length=200), nullable=True))
    if "professional_role" not in columns:
        op.add_column("users", sa.Column("professional_role", sa.String(length=48), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("users"):
        return
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "professional_role" in columns:
        op.drop_column("users", "professional_role")
    if "organization" in columns:
        op.drop_column("users", "organization")
