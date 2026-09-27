from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

from app.core.config import settings
from app.core.logging import logger

# Dense vector with the CLIP image embedding of the cropped car.
CLIP_FIELD = "clip_embedding"

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

    async def ensure_text_search_mapping(self) -> bool:
        """
        Add the ``clip_embedding`` field to an existing cars index.

        Adding a new field is a non-destructive mapping update, so indices
        created before text search was introduced keep their data. Returns
        True if the field is present after the call.
        """
        client = self._ensure_connected()
        index = settings.ELASTICSEARCH_INDEX
        if not await client.indices.exists(index=index):
            logger.warning(
                "Index '%s' does not exist yet; clip_embedding mapping will be "
                "created by the es-setup job",
                index,
            )
            return False

        mapping = await client.indices.get_mapping(index=index)
        properties = (
            mapping.get(index, {}).get("mappings", {}).get("properties", {})
        )
        if CLIP_FIELD in properties:
            return True

        await client.indices.put_mapping(
            index=index,
            properties={
                CLIP_FIELD: {
                    "type": "dense_vector",
                    "dims": settings.NEURON5_CLIP_EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                }
            },
        )
        logger.info("Added '%s' field to index '%s'", CLIP_FIELD, index)
        return True

    @staticmethod
    def _plate_filter(plate_number: str) -> dict[str, Any]:
        return {
            "wildcard": {
                "plate_number": {
                    "value": plate_number,
                    "case_insensitive": True,
                }
            }
        }

    async def index_car(
        self,
        car_id: str,
        plate_number: str,
        embedding: list[float],
        s3_key: str,
        metadata: dict[str, Any] | None = None,
        clip_embedding: list[float] | None = None,
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
            if clip_embedding:
                document[CLIP_FIELD] = clip_embedding

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
        top_k: int = 5,
        text_embedding: list[float] | None = None,
        text_weight: float = 0.5,
    ) -> dict[str, Any]:
        """
        Search for cars by plate number filter and embedding similarity.

        First filters by plate number (partial match), then ranks by
        embedding similarity to return top_k results.

        If ``text_embedding`` is given (photo + description search), a second
        kNN clause over ``clip_embedding`` is added; Elasticsearch sums the
        boosted scores of both clauses.

        Args:
            plate_number: License plate number to filter by
            query_embedding: Query embedding vector
            top_k: Number of results to return
            text_embedding: Optional CLIP text embedding of a description
            text_weight: Share of the text clause in the final score (0..1)

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
                knn["filter"] = self._plate_filter(plate_number)

            knn_clauses: dict[str, Any] | list[dict[str, Any]] = knn
            if text_embedding:
                text_weight = min(max(text_weight, 0.0), 1.0)
                text_knn = {
                    "field": CLIP_FIELD,
                    "query_vector": text_embedding,
                    "k": top_k,
                    "num_candidates": top_k * 10,
                    "boost": text_weight,
                }
                if plate_number:
                    text_knn["filter"] = self._plate_filter(plate_number)
                knn["boost"] = 1.0 - text_weight
                knn_clauses = [knn, text_knn]

            search_body = {
                "knn": knn_clauses,
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

    async def search_by_text_embedding(
        self,
        text_embedding: list[float],
        plate_number: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Search cars by a CLIP text embedding of a free-text description,
        optionally restricted by a plate wildcard.
        """
        knn: dict[str, Any] = {
            "field": CLIP_FIELD,
            "query_vector": text_embedding,
            "k": top_k,
            "num_candidates": max(top_k * 20, 100),
        }
        if plate_number:
            knn["filter"] = self._plate_filter(plate_number)

        client = self._ensure_connected()
        response = await client.search(
            index=settings.ELASTICSEARCH_INDEX,
            knn=knn,
            size=top_k,
            source_excludes=["embedding", CLIP_FIELD],
        )
        hits = response.get("hits", {})
        total_found = hits.get("total", {}).get("value", 0)
        logger.info(
            "Text search: %d hits for plate filter '%s'",
            total_found,
            plate_number or "<none>",
        )
        return {"hits": hits.get("hits", []), "total": total_found}

    async def search_by_plate(
        self, plate_number: str, top_k: int = 5
    ) -> dict[str, Any]:
        """Search cars by plate wildcard only (no visual ranking)."""
        client = self._ensure_connected()
        response = await client.search(
            index=settings.ELASTICSEARCH_INDEX,
            query={"constant_score": {"filter": self._plate_filter(plate_number)}},
            size=top_k,
            track_total_hits=True,
            source_excludes=["embedding", CLIP_FIELD],
        )
        hits = response.get("hits", {})
        total_found = hits.get("total", {}).get("value", 0)
        logger.info("Plate search: %d hits for '%s'", total_found, plate_number)
        return {"hits": hits.get("hits", []), "total": total_found}

    async def get_cars_missing_clip_embedding(
        self, size: int, exclude_ids: list[str] | None = None
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (cars without clip_embedding, total count of such cars)."""
        query: dict[str, Any] = {
            "bool": {"must_not": [{"exists": {"field": CLIP_FIELD}}]}
        }
        if exclude_ids:
            query["bool"]["must_not"].append({"ids": {"values": exclude_ids}})

        client = self._ensure_connected()
        response = await client.search(
            index=settings.ELASTICSEARCH_INDEX,
            query=query,
            size=size,
            track_total_hits=True,
            source_includes=["car_id", "s3_key", "plate_number"],
        )
        hits = response.get("hits", {})
        cars = [
            {**hit.get("_source", {}), "_id": hit.get("_id")}
            for hit in hits.get("hits", [])
        ]
        return cars, hits.get("total", {}).get("value", 0)

    async def update_clip_embedding(
        self, doc_id: str, clip_embedding: list[float]
    ) -> bool:
        client = self._ensure_connected()
        response = await client.update(
            index=settings.ELASTICSEARCH_INDEX,
            id=doc_id,
            doc={CLIP_FIELD: clip_embedding},
        )
        return response.get("result") in ("updated", "noop")

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
