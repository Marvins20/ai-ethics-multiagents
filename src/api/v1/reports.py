from fastapi import APIRouter, Depends, Query
from ...api.deps import get_current_user
from ...db.models.user import User
from ...services.incidents_reports_etl_service import get_reports_by_report_numbers

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("", response_model=list[dict])
async def get_reports(
    ids: str = Query(..., description="Comma-separated list of report_number values"),
    _: User = Depends(get_current_user),
):
    report_numbers = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    return get_reports_by_report_numbers(report_numbers)
