from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.services.s3_client import s3_client

router = APIRouter(prefix="/images", tags=["images"])


@router.get("/{object_key:path}")
async def get_image(object_key: str) -> Response:
    image_data = await s3_client.download_image(object_key)
    if image_data is None:
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = "image/png" if object_key.lower().endswith(".png") else "image/jpeg"
    return Response(content=image_data, media_type=media_type)
