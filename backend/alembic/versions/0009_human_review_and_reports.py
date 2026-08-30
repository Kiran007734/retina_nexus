"""Add clinician modification metadata and review decision values."""

import sqlalchemy as sa
from alembic import op


revision = "0009_human_review_and_reports"
down_revision = "0008_screening_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("clinical_reviews"):
        columns = {column["name"] for column in inspector.get_columns("clinical_reviews")}
        if "modified_grade" not in columns:
            op.add_column("clinical_reviews", sa.Column("modified_grade", sa.Integer(), nullable=True))
    if bind.dialect.name == "postgresql":
        for value in ("approve", "modify", "reject", "request_recapture"):
            op.execute(sa.text(f"ALTER TYPE reviewdecision ADD VALUE IF NOT EXISTS '{value}'"))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("clinical_reviews"):
        columns = {column["name"] for column in inspector.get_columns("clinical_reviews")}
        if "modified_grade" in columns:
            op.drop_column("clinical_reviews", "modified_grade")
