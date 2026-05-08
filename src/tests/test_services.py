import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestS3Client:
    """Tests for S3 client."""

    def test_ensure_connected_raises_when_not_initialized(self):
        """Test that _ensure_connected raises if client is None."""
        from app.services.s3_client import S3Client

        client = S3Client()
        client.client = None

        with pytest.raises(RuntimeError, match="not connected"):
            client._ensure_connected()

    @patch("app.services.s3_client.boto3")
    def test_connect_initializes_client(self, mock_boto3):
        """Test that connect initializes the boto3 client."""
        from app.services.s3_client import S3Client

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        client = S3Client()
        client.connect()

        assert client.client is not None
        mock_boto3.client.assert_called_once()


class TestElasticsearchClient:
    """Tests for Elasticsearch client."""

    def test_ensure_connected_raises_when_not_initialized(self):
        """Test that _ensure_connected raises if client is None."""
        from app.services.elasticsearch_client import ElasticsearchClient

        client = ElasticsearchClient()
        client.client = None

        with pytest.raises(RuntimeError, match="not connected"):
            client._ensure_connected()
