from fastapi import Request
from fastapi.responses import JSONResponse
from jose import JWTError


async def jwt_exception_handler(request: Request, exc: JWTError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})
