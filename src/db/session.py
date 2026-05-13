from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from .base import Base
from ..config import settings

if "asyncpg" in settings.database_url:
    _connect_args: dict = {"statement_cache_size": 0}
elif "psycopg" in settings.database_url:
    _connect_args = {"prepare_threshold": 0}
else:
    _connect_args = {}

engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:  # type: ignore[return]
    async with AsyncSessionLocal() as session:
        yield session
