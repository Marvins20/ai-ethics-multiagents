import uuid
from datetime import datetime
from pydantic import BaseModel


class HasReadResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    norm_id: uuid.UUID
    read_at: datetime

    model_config = {"from_attributes": True}
