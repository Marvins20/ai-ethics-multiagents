import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_current_user, require_admin, get_db
from ...db.models.user import User
from ...db.models.project_type import ProjectType
from ...schemas.project_type import ProjectTypeCreate, ProjectTypeUpdate, ProjectTypeResponse

router = APIRouter(prefix="/project-types", tags=["project-types"])


@router.post("", response_model=ProjectTypeResponse, status_code=status.HTTP_201_CREATED)
async def create_project_type(
    body: ProjectTypeCreate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    pt = ProjectType(nome=body.nome)
    db.add(pt)
    await db.commit()
    await db.refresh(pt)
    return pt


@router.get("", response_model=list[ProjectTypeResponse])
async def list_project_types(
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ProjectType))
    return result.scalars().all()


@router.get("/{project_type_id}", response_model=ProjectTypeResponse)
async def get_project_type(
    project_type_id: uuid.UUID,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ProjectType).where(ProjectType.id == project_type_id))
    pt = result.scalar_one_or_none()
    if pt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project type not found")
    return pt


@router.patch("/{project_type_id}", response_model=ProjectTypeResponse)
async def update_project_type(
    project_type_id: uuid.UUID,
    body: ProjectTypeUpdate,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ProjectType).where(ProjectType.id == project_type_id))
    pt = result.scalar_one_or_none()
    if pt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project type not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(pt, field, value)
    await db.commit()
    await db.refresh(pt)
    return pt


@router.delete("/{project_type_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_type(
    project_type_id: uuid.UUID,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(ProjectType).where(ProjectType.id == project_type_id))
    pt = result.scalar_one_or_none()
    if pt is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project type not found")
    await db.delete(pt)
    await db.commit()
