from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import datetime

from app.monitoring.metrics import feedback_actions_total
from app.services.inference_logger import log_feedback_event

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    result_id: str = Field(..., description="car_id or search result identifier")
    action: str = Field(..., description="User action: confirm, reject, correct")
    corrected_plate: Optional[str] = None
    comment: Optional[str] = None
    disputed: bool = False  # пометка спорного кейса


@router.post("", summary="Submit user feedback",
             description="Record user feedback on model results (confirm, reject, correct).")
async def submit_feedback(feedback: FeedbackRequest):
    if feedback.action not in ("confirm", "reject", "correct"):
        raise HTTPException(status_code=400, detail="Invalid action")

    # Увеличиваем счётчик действий
    feedback_actions_total.labels(action=feedback.action).inc()
    if feedback.disputed:
        feedback_actions_total.labels(action="disputed").inc()

    # Логируем событие в Elasticsearch
    await log_feedback_event({
        "result_id": feedback.result_id,
        "action": feedback.action,
        "corrected_plate": feedback.corrected_plate,
        "comment": feedback.comment,
        "disputed": feedback.disputed,
        "@timestamp": datetime.datetime.utcnow().isoformat()
    })

    return {"status": "ok"}