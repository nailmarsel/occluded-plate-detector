import datetime

from app.services.elasticsearch_client import es_client


async def log_inference_event(event: dict):
    """
    Log an inference event to Elasticsearch index 'inference-logs'.
    event keys: timestamp, endpoint, plate_number, plate_length,
                car_confidence, plate_confidence, ocr_confidence,
                image_size_bytes, success, error_stage, fallback_used
    """
    event["@timestamp"] = datetime.datetime.utcnow().isoformat()
    await es_client.index_document(
        index="inference-logs",
        body=event
    )


async def log_feedback_event(event: dict):
    await es_client.index_document(index="feedback-logs", body=event)
