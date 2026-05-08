#!/bin/bash
# Setup script for Elasticsearch index mapping
# Run this after starting docker-compose

set -e

ES_HOST="http://localhost:9200"
ES_USER="elastic"
ES_PASS="changeme"
ES_INDEX="cars"

echo "Setting up Elasticsearch index mapping..."

curl -X PUT "${ES_HOST}/${ES_INDEX}" \
  -u "${ES_USER}:${ES_PASS}" \
  -H "Content-Type: application/json" \
  -d '{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "car_id": {
        "type": "keyword"
      },
      "plate_number": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 512,
        "index": true,
        "similarity": "cosine"
      },
      "s3_key": {
        "type": "keyword"
      },
      "metadata": {
        "type": "object",
        "enabled": false
      },
      "created_at": {
        "type": "date",
        "format": "epoch_millis"
      }
    }
  }
}'

echo ""
echo "Elasticsearch index mapping created successfully!"
echo "Index: ${ES_INDEX}"
echo "Embedding dims: 512"
echo "Similarity: cosine"
