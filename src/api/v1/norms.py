import uuid
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

UPLOADS_DIR = Path("data/uploads/norms")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}

from ...api.deps import get_current_user, require_admin, get_db
from ...db.models.user import User
from ...db.models.norm import Norm
from ...db.models.has_read import HasRead
from ...db.enums import PrivilegioEnum
from ...schemas.norm import NormCreate, NormUpdate, NormResponse
from ...schemas.has_read import HasReadResponse

router = APIRouter(prefix="/norms", tags=["norms"])


@router.post("", response_model=NormResponse, status_code=status.HTTP_201_CREATED)
async def create_norm(
    body: NormCreate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    norm = Norm(**body.model_dump())
    db.add(norm)
    await db.commit()
    await db.refresh(norm)
    return norm


@router.get("", response_model=list[NormResponse])
async def list_norms(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.privilegio == PrivilegioEnum.admin:
        result = await db.execute(select(Norm))
    else:
        result = await db.execute(select(Norm).where(Norm.publico == True))  # noqa: E712
    return result.scalars().all()


@router.get("/{norm_id}", response_model=NormResponse)
async def get_norm(
    norm_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    norm = await _get_norm_or_404(norm_id, db)
    if not norm.publico and current_user.privilegio != PrivilegioEnum.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return norm


@router.patch("/{norm_id}", response_model=NormResponse)
async def update_norm(
    norm_id: uuid.UUID,
    body: NormUpdate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    norm = await _get_norm_or_404(norm_id, db)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(norm, field, value)
    await db.commit()
    await db.refresh(norm)
    return norm


@router.delete("/{norm_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_norm(
    norm_id: uuid.UUID,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    norm = await _get_norm_or_404(norm_id, db)
    await db.delete(norm)
    await db.commit()


@router.post("/{norm_id}/upload", response_model=NormResponse)
async def upload_norm_file(
    norm_id: uuid.UUID,
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    norm = await _get_norm_or_404(norm_id, db)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Tipo de arquivo não suportado. Use: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    dest = UPLOADS_DIR / f"{norm_id}{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    norm.arquivo = f"/uploads/norms/{norm_id}{ext}"
    await db.commit()
    await db.refresh(norm)

    if norm.incluir_avaliacao:
        try:
            from ...services.proprietary_framework_etl_service import ingest_norm_document
            ingest_norm_document(str(dest), str(norm_id), norm.titulo)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Failed to ingest norm %s: %s", norm_id, exc)

    return norm

@router.post("/{norm_id}/read", response_model=HasReadResponse, status_code=status.HTTP_201_CREATED)
async def mark_as_read(
    norm_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _get_norm_or_404(norm_id, db)

    existing = await db.execute(
        select(HasRead).where(
            HasRead.user_id == current_user.id,
            HasRead.norm_id == norm_id,
        )
    )
    record = existing.scalar_one_or_none()
    if record is not None:
        return record

    record = HasRead(
        user_id=current_user.id,
        norm_id=norm_id,
        read_at=datetime.now(timezone.utc),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


@router.get("/{norm_id}/readers", response_model=list[HasReadResponse])
async def list_readers(
    norm_id: uuid.UUID,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    await _get_norm_or_404(norm_id, db)
    result = await db.execute(select(HasRead).where(HasRead.norm_id == norm_id))
    return result.scalars().all()


@router.get("/me/read", response_model=list[HasReadResponse])
async def my_read_norms(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(HasRead).where(HasRead.user_id == current_user.id)
    )
    return result.scalars().all()


async def _get_norm_or_404(norm_id: uuid.UUID, db: AsyncSession) -> Norm:
    result = await db.execute(select(Norm).where(Norm.id == norm_id))
    norm = result.scalar_one_or_none()
    if norm is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Norm not found")
    return norm
