import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..base import Base


class Norm(Base):
    __tablename__ = "norms"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    titulo: Mapped[str] = mapped_column(String(500), nullable=False)

    # Relative path "norms/uuid_filename.pdf"
    arquivo: Mapped[str | None] = mapped_column(String(1000), nullable=True)


    publico: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    incluir_avaliacao: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    obrigatorio: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    has_reads: Mapped[list["HasRead"]] = relationship(back_populates="norm", cascade="all, delete-orphan")  # type: ignore[name-defined]
