"""Worker tasks package — re-exports all task functions for backward compatibility."""
from .decompose import run_decompose_task
from .audit import run_audit_task
from .questionnaire import run_questionnaire_task
from .embedding import run_user_embedding_task
from .pairing import run_pairing_task

__all__ = [
    "run_decompose_task",
    "run_audit_task",
    "run_questionnaire_task",
    "run_user_embedding_task",
    "run_pairing_task",
]
