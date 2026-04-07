"""
Enqueue embedding jobs for every user that has no stored embedding.
Run with: uv run python scripts/backfill_embeddings.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy import select, not_, exists

from src.config import settings
from src.db.session import AsyncSessionLocal
from src.db.models.user import User
from src.db.models.user_embedding import UserEmbedding


async def main() -> None:
    async with AsyncSessionLocal() as db:
        rows = await db.execute(
            select(User).where(
                not_(exists().where(UserEmbedding.user_id == User.id))
            )
        )
        users = rows.scalars().all()

    if not users:
        print("All users already have embeddings.")
        return

    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    for u in users:
        await redis_pool.enqueue_job("run_user_embedding_task", str(u.id))
        print(f"  Enqueued: {u.email} ({u.id})")
    await redis_pool.aclose()

    print(f"\nDone — {len(users)} job(s) enqueued. Make sure the ARQ worker is running.")


if __name__ == "__main__":
    asyncio.run(main())
