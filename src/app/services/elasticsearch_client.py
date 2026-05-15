from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

from app.core.config import settings
from app.core.logging import logger


class ElasticsearchClient:
    """Async Elasticsearch client for car data operations."""

    def __init__(self):
        self.client: AsyncElasticsearch | None = None

    def _ensure_connected(self) -> AsyncElasticsearch:
        """Ensure the client is connected, raise otherwise."""
        if self.client is None:
            raise RuntimeError(
                "Elasticsearch client is not connected. Call connect() first."
            )
        return self.client

    async def connect(self) -> None:
        """Establish connection to Elasticsearch."""
        try:
            basic_auth = None
            if settings.ELASTICSEARCH_USERNAME or settings.ELASTICSEARCH_PASSWORD:
                basic_auth = (
                    settings.ELASTICSEARCH_USERNAME,
                    settings.ELASTICSEARCH_PASSWORD,
                )

            self.client = AsyncElasticsearch(
                hosts=[{
                    "host": settings.ELASTICSEARCH_HOST,
                    "port": settings.ELASTICSEARCH_PORT,
                    "scheme": "http"
                }],
                basic_auth=basic_auth,
            )
            # Test connection
            client = self._ensure_connected()
            info = await client.info()
            logger.info(f"Connected to Elasticsearch: {info['version']['number']}")
        except ConnectionError as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Elasticsearch: {e}")
            raise

    async def close(self) -> None:
        """Close Elasticsearch connection."""
        if self.client:
            await self.client.close()
            logger.info("Elasticsearch connection closed")

    async def check_health(self) -> bool:
        """Check if Elasticsearch is reachable."""
        try:
            if self.client:
                await self.client.ping()
                return True
            return False
        except Exception:
            return False

    async def index_document(self, index: str, body: dict):
        return await self.client.index(index=index, body=body)

    async def index_car(
        self,
        car_id: str,
        plate_number: str,
        embedding: list[float],
        s3_key: str,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Index a car document in Elasticsearch.

        Args:
            car_id: Unique identifier for the car
            plate_number: License plate number
            embedding: Car image embedding vector
            s3_key: S3 object key for the car image
            metadata: Additional metadata

        Returns:
            True if indexing succeeded
        """
        try:
            document = {
                "car_id": car_id,
                "plate_number": plate_number,
                "embedding": embedding,
                "s3_key": s3_key,
                "metadata": metadata or {},
                "created_at": None,  # Will be set by Elasticsearch
            }

            client = self._ensure_connected()
            response = await client.index(
                index=settings.ELASTICSEARCH_INDEX,
                id=car_id,
                document=document
            )
            logger.info(f"Indexed car {car_id} in Elasticsearch with S3 key: {s3_key}")
            return response.get("result") in ("created", "updated")
        except Exception as e:
            logger.error(f"Failed to index car {car_id}: {e}")
            raise

    async def search_by_plate_and_embedding(
        self,
        plate_number: str,
        query_embedding: list[float],
        top_k: int = 5
    ) -> dict[str, Any]:
        """
        Search for cars by plate number filter and embedding similarity.

        First filters by plate number (partial match), then ranks by
        embedding similarity to return top_k results.

        Args:
            plate_number: License plate number to filter by
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            dict with hits and metadata
        """
        try:
            knn = {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 10,
            }

            if plate_number:
                knn["filter"] = {
                    "wildcard": {
                        "plate_number": {
                            "value": plate_number,
                            "case_insensitive": True,
                        }
                    }
                }

            search_body = {
                "knn": knn,
                "size": top_k,
            }

            client = self._ensure_connected()
            response = await client.search(
                index=settings.ELASTICSEARCH_INDEX,
                body=search_body
            )

            hits = response.get("hits", {})
            total_found = hits.get("total", {}).get("value", 0)

            logger.info(
                f"Found {total_found} cars for plate filter "
                f"'{plate_number or '<none>'}', returning top {top_k} "
                "by embedding similarity"
            )

            return {
                "hits": hits.get("hits", []),
                "total": total_found
            }
        except Exception as e:
            logger.error(f"Search failed for plate '{plate_number}': {e}")
            raise

    async def get_car_by_id(self, car_id: str) -> dict[str, Any] | None:
        """Retrieve a car document by ID."""
        try:
            client = self._ensure_connected()
            response = await client.get(
                index=settings.ELASTICSEARCH_INDEX,
                id=car_id
            )
            return response.get("_source")
        except NotFoundError:
            logger.warning(f"Car {car_id} not found in Elasticsearch")
            return None
        except Exception as e:
            logger.error(f"Failed to get car {car_id}: {e}")
            raise

    async def get_car_by_plate(self, plate_number: str) -> dict[str, Any] | None:
        """Retrieve the first car document with an exact plate number."""
        try:
            client = self._ensure_connected()
            response = await client.search(
                index=settings.ELASTICSEARCH_INDEX,
                body={
                    "query": {
                        "term": {
                            "plate_number": plate_number,
                        }
                    },
                    "size": 1,
                },
            )
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                return None

            source = hits[0].get("_source", {})
            source["_id"] = hits[0].get("_id")
            return source
        except Exception as e:
            logger.error(f"Failed to get car by plate {plate_number}: {e}")
            raise

    async def delete_car(self, car_id: str) -> bool:
        """Delete a car document by ID."""
        try:
            client = self._ensure_connected()
            response = await client.delete(
                index=settings.ELASTICSEARCH_INDEX,
                id=car_id
            )
            logger.info(f"Deleted car {car_id} from Elasticsearch")
            return response.get("result") == "deleted"
        except NotFoundError:
            logger.warning(f"Car {car_id} not found for deletion")
            return False
        except Exception as e:
            logger.error(f"Failed to delete car {car_id}: {e}")
            raise


# Global Elasticsearch client instance
es_client = ElasticsearchClient()
