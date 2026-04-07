import uuid
from datetime import datetime, timezone
from sqlalchemy import DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..base import Base


class HasRead(Base):
    __tablename__ = "has_read"
    __table_args__ = (
        UniqueConstraint("user_id", "norm_id", name="uq_has_read_user_norm"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    norm_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("norms.id", ondelete="CASCADE"), nullable=False
    )
    read_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped["User"] = relationship(back_populates="has_reads")  # type: ignore[name-defined]
    norm: Mapped["Norm"] = relationship(back_populates="has_reads")  # type: ignore[name-defined]
