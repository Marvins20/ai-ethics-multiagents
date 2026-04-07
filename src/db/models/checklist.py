import uuid
from sqlalchemy import String, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..base import Base


class Checklist(Base):
    __tablename__ = "checklists"
    __table_args__ = (
        UniqueConstraint("project_type_id", name="uq_checklist_project_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    nome: Mapped[str] = mapped_column(String(255), nullable=False)
    descricao: Mapped[str | None] = mapped_column(Text, nullable=True)

    project_type_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("project_types.id", ondelete="CASCADE"), nullable=False
    )

    project_type: Mapped["ProjectType"] = relationship(back_populates="checklist")  # type: ignore[name-defined]
    groups: Mapped[list["QuestionGroup"]] = relationship(  # type: ignore[name-defined]
        back_populates="checklist", cascade="all, delete-orphan", order_by="QuestionGroup.ordem"
    )
    responses: Mapped[list["ChecklistResponse"]] = relationship(  # type: ignore[name-defined]
        back_populates="checklist", cascade="all, delete-orphan"
    )


class QuestionGroup(Base):
    __tablename__ = "question_groups"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    checklist_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("checklists.id", ondelete="CASCADE"), nullable=False
    )
    nome: Mapped[str] = mapped_column(String(255), nullable=False)
    ordem: Mapped[int] = mapped_column(nullable=False, default=0)

    checklist: Mapped["Checklist"] = relationship(back_populates="groups")
    questions: Mapped[list["Question"]] = relationship(  # type: ignore[name-defined]
        back_populates="group", cascade="all, delete-orphan", order_by="Question.ordem"
    )


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("question_groups.id", ondelete="CASCADE"), nullable=False
    )
    descricao: Mapped[str] = mapped_column(Text, nullable=False)
    ordem: Mapped[int] = mapped_column(nullable=False, default=0)

    group: Mapped["QuestionGroup"] = relationship(back_populates="questions")
    responses: Mapped[list["ChecklistResponse"]] = relationship(  # type: ignore[name-defined]
        back_populates="question", cascade="all, delete-orphan"
    )
