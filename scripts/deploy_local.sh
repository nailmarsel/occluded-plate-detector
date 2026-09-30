#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"

for command in docker curl python3; do
    command -v "$command" >/dev/null || { printf 'Required command missing: %s\n' "$command" >&2; exit 1; }
done

[[ -f "$SRC_DIR/.env" ]] || { printf 'Missing src/.env\n' >&2; exit 1; }
for model in car_detector license_plate_detector plate_ocr car_embedder; do
    [[ -s "$SRC_DIR/models/$model.pt" ]] || {
        printf 'Missing or empty model: src/models/%s.pt\n' "$model" >&2
        exit 1
    }
done

compose=(docker compose --project-directory "$SRC_DIR" --env-file "$SRC_DIR/.env"
    -f "$SRC_DIR/docker-compose.yml" -f "$SRC_DIR/docker-compose.local.yml" --profile full)
"${compose[@]}" config --quiet
# CPU model initialization and infrastructure startup can take several minutes.
"${compose[@]}" up -d --build --wait --wait-timeout 600

curl --noproxy '*' --fail --silent --show-error --retry 12 --retry-connrefused \
    --retry-delay 5 --retry-max-time 120 --connect-timeout 5 --max-time 15 \
    http://localhost/app/ --output /dev/null
curl --noproxy '*' --fail --silent --show-error --retry 12 --retry-connrefused \
    --retry-delay 5 --retry-max-time 120 --connect-timeout 5 --max-time 15 \
    http://localhost/api/v1/health | python3 -c '
import json, sys
payload = json.load(sys.stdin)
if payload.get("status") != "healthy":
    sys.exit("API health status is not healthy")
'
printf 'Local deployment ready: http://localhost/app/\n'
