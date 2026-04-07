import uuid
from datetime import datetime, timezone
from sqlalchemy import Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..base import Base


class ChecklistResponse(Base):
    __tablename__ = "checklist_responses"
    __table_args__ = (
        # A user can only answer a given question once per project
        UniqueConstraint("project_id", "question_id", "user_id", name="uq_response_per_user_project_question"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    checklist_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("checklists.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    resposta: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    checklist: Mapped["Checklist"] = relationship(back_populates="responses")  # type: ignore[name-defined]
    project: Mapped["Project"] = relationship(back_populates="checklist_responses")  # type: ignore[name-defined]
    question: Mapped["Question"] = relationship(back_populates="responses")  # type: ignore[name-defined]
    user: Mapped["User"] = relationship(back_populates="checklist_responses")  # type: ignore[name-defined]
