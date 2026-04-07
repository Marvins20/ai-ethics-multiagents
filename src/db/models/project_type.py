import uuid
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..base import Base


class ProjectType(Base):
    __tablename__ = "project_types"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    nome: Mapped[str] = mapped_column(String(255), nullable=False)

    checklist: Mapped["Checklist | None"] = relationship(back_populates="project_type", uselist=False, cascade="all, delete-orphan", passive_deletes=True)  # type: ignore[name-defined]
    projects: Mapped[list["Project"]] = relationship(back_populates="project_type")  # type: ignore[name-defined]
