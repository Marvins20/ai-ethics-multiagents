from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from .base import Base
from ..config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={"statement_cache_size": 0},
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:  # type: ignore[return]
    async with AsyncSessionLocal() as session:
        yield session
