import uuid
from datetime import datetime, timezone
from sqlalchemy import ForeignKey, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..base import Base


class ProjectEvaluatorAssignment(Base):
    __tablename__ = "project_evaluator_assignments"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    evaluator_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    load_at_assignment: Mapped[int] = mapped_column(nullable=False, default=0)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    project: Mapped["Project"] = relationship(back_populates="evaluator_assignments")  # type: ignore[name-defined]
    evaluator: Mapped["User"] = relationship(back_populates="evaluator_assignments")  # type: ignore[name-defined]
