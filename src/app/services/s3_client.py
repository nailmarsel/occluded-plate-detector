import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config
from typing import Optional, Any
from app.core.config import settings
from app.core.logging import logger


class S3Client:
    """S3/Garage client for image storage."""

    def __init__(self):
        self.client: Optional[Any] = None
        self.bucket_name = settings.S3_BUCKET_NAME

    def _ensure_connected(self) -> Any:
        """Ensure the client is connected, raise otherwise."""
        if self.client is None:
            raise RuntimeError("S3 client is not connected. Call connect() first.")
        return self.client

    def connect(self) -> None:
        """Initialize S3 client and ensure bucket exists."""
        try:
            self.client = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
                region_name=settings.S3_REGION,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                ),
                use_ssl=settings.S3_USE_SSL,
            )

            # Test connection and create bucket if not exists
            self._ensure_bucket_exists()
            logger.info(
                f"S3 client initialized: {settings.S3_ENDPOINT_URL}/{self.bucket_name}"
            )
        except NoCredentialsError:
            logger.error("S3 credentials not configured")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def close(self) -> None:
        """Close S3 client connection."""
        if self.client:
            self.client.close()
            logger.info("S3 client connection closed")

    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        client = self._ensure_connected()
        try:
            client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' already exists")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchBucket"):
                logger.info(f"Creating bucket '{self.bucket_name}'")
                client.create_bucket(Bucket=self.bucket_name)
                logger.info(f"Bucket '{self.bucket_name}' created")
            else:
                logger.error(f"Error checking bucket: {e}")
                raise

    async def upload_image(
        self,
        image_data: bytes,
        object_key: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """
        Upload image to S3/Garage.

        Args:
            image_data: Raw image bytes
            object_key: S3 object key (e.g., "cars/car_001.jpg")
            content_type: MIME type of the image

        Returns:
            S3 object key if successful
        """
        try:
            client = self._ensure_connected()
            client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=image_data,
                ContentType=content_type,
            )
            logger.info(f"Uploaded image: {object_key}")
            return object_key
        except ClientError as e:
            logger.error(f"Failed to upload image {object_key}: {e}")
            raise

    async def download_image(self, object_key: str) -> Optional[bytes]:
        """
        Download image from S3/Garage.

        Args:
            object_key: S3 object key

        Returns:
            Raw image bytes or None if not found
        """
        try:
            client = self._ensure_connected()
            response = client.get_object(
                Bucket=self.bucket_name, Key=object_key
            )
            image_data = response["Body"].read()
            logger.info(f"Downloaded image: {object_key}")
            return image_data
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "NoSuchKey":
                logger.warning(f"Image not found: {object_key}")
                return None
            logger.error(f"Failed to download image {object_key}: {e}")
            raise

    async def delete_image(self, object_key: str) -> bool:
        """
        Delete image from S3/Garage.

        Args:
            object_key: S3 object key

        Returns:
            True if deleted successfully
        """
        try:
            client = self._ensure_connected()
            client.delete_object(Bucket=self.bucket_name, Key=object_key)
            logger.info(f"Deleted image: {object_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete image {object_key}: {e}")
            raise

    def get_presigned_url(
        self, object_key: str, expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access.

        Args:
            object_key: S3 object key
            expiration: URL expiration in seconds

        Returns:
            Presigned URL or None if generation failed
        """
        try:
            client = self._ensure_connected()
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {object_key}: {e}")
            return None

    def get_object_url(self, object_key: str) -> str:
        """
        Get the full URL to an S3 object.

        Args:
            object_key: S3 object key

        Returns:
            Full URL to the object
        """
        return f"{settings.S3_ENDPOINT_URL}/{self.bucket_name}/{object_key}"


# Global S3 client instance
s3_client = S3Client()
