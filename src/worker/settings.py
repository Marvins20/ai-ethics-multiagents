from arq.connections import RedisSettings

from ..config import settings
from .tasks import (
    run_audit_task,
    run_decompose_task,
    run_questionnaire_task,
    run_user_embedding_task,
    run_pairing_task,
)


class WorkerSettings:
    functions = [
        run_audit_task,
        run_decompose_task,
        run_questionnaire_task,
        run_user_embedding_task,
        run_pairing_task,
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 5
    job_timeout = 600  
