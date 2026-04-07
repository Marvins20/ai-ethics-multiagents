from fastapi import APIRouter, Depends
from ..deps import verify_api_key
from .auth import router as auth_router
from .users import router as users_router
from .projects import router as projects_router
from .audits import router as audits_router
from .project_types import router as project_types_router
from .checklists import router as checklists_router
from .checklist_responses import router as checklist_responses_router
from .norms import router as norms_router
from .reports import router as reports_router
from .questionnaires import router as questionnaires_router
from .pairing import router as pairing_router

api_router = APIRouter(dependencies=[Depends(verify_api_key)])
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(projects_router)
api_router.include_router(audits_router)
api_router.include_router(project_types_router)
api_router.include_router(checklists_router)
api_router.include_router(checklist_responses_router)
api_router.include_router(norms_router)
api_router.include_router(reports_router)
api_router.include_router(questionnaires_router)
api_router.include_router(pairing_router)
