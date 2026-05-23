#!/usr/bin/env python3
"""Export operator feedback into training data — AutobahnCV.

This is step 1 of the retraining cycle (``scripts/retrain.py``). It pulls
operator feedback from the Elasticsearch ``feedback-logs`` index, resolves the
referenced car records, and writes per-model training/review files that the
``ml/<workspace>`` preparation scripts can consume.

Index schema (written by ``src/app/services/inference_logger.py``):

    feedback-logs : result_id, action, corrected_plate, comment, disputed, @timestamp
    cars          : car_id, plate_number, embedding, s3_key, metadata, created_at

How each model uses feedback:

    plate_ocr      -> ``correct`` events: ``corrected_plate`` is the true label
    car_embedder   -> ``confirm`` / ``correct`` = positive identity,
                      ``reject`` = hard negative
    car_detector,  -> ``reject`` / ``disputed`` events exported as hard cases
    plate_detector    for manual re-labelling (detection has no direct label
                      in feedback)

Usage:
    # from a live Elasticsearch instance
    python3 scripts/export_feedback.py --model plate_ocr --since-days 30

    # offline, from a dumped feedback-logs export (no Elasticsearch needed)
    python3 scripts/export_feedback.py --model plate_ocr --from-file dump.json

    # inspect only, write nothing
    python3 scripts/export_feedback.py --model plate_ocr --dry-run

Elasticsearch connection defaults come from the same environment variables the
app uses (ELASTICSEARCH_HOST / ELASTICSEARCH_PORT / ELASTICSEARCH_USERNAME /
ELASTICSEARCH_PASSWORD) and can be overridden with CLI flags.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FEEDBACK_INDEX = "feedback-logs"

# Per-model workspace + which feedback actions are relevant.
WORKSPACES: dict[str, dict[str, object]] = {
    "plate_ocr": {
        "dir": "ml/plate_ocr/data/feedback",
        "label_actions": ["correct"],          # corrected_plate is a real label
        "review_actions": ["reject", "disputed"],
    },
    "car_embedder": {
        "dir": "ml/car_embedder/data/feedback",
        "label_actions": ["confirm", "correct"],  # confirmed identity
        "review_actions": ["reject", "disputed"],
    },
    "car_detector": {
        "dir": "ml/car_detector/data/feedback",
        "label_actions": [],                   # no direct detection label
        "review_actions": ["reject", "disputed"],
    },
    "plate_detector": {
        "dir": "ml/plate_detector/data/feedback",
        "label_actions": [],
        "review_actions": ["reject", "disputed"],
    },
}


# --------------------------------------------------------------------------
# Feedback sources: live Elasticsearch, or an offline JSON dump.
# --------------------------------------------------------------------------
def fetch_from_es(args: argparse.Namespace) -> tuple[list[dict], object]:
    """Return (feedback_events, es_client). es_client is None for offline mode."""
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        print("ERROR: the 'elasticsearch' package is not installed.", file=sys.stderr)
        print("Install it (pip install elasticsearch) or use --from-file.", file=sys.stderr)
        raise SystemExit(2)

    scheme = "https" if args.es_ssl else "http"
    auth = None
    if args.es_user or args.es_password:
        auth = (args.es_user, args.es_password)

    es = Elasticsearch(
        hosts=[f"{scheme}://{args.es_host}:{args.es_port}"],
        basic_auth=auth,
        request_timeout=15,
    )
    try:
        if not es.ping():
            raise ConnectionError("ping failed")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot reach Elasticsearch at {args.es_host}:{args.es_port}: {exc}",
              file=sys.stderr)
        print("Start the stack or use --from-file with a dumped export.", file=sys.stderr)
        raise SystemExit(2)

    since = (datetime.now(timezone.utc) - timedelta(days=args.since_days)).isoformat()
    query = {
        "bool": {
            "filter": [{"range": {"@timestamp": {"gte": since}}}],
        }
    }
    resp = es.search(index=FEEDBACK_INDEX, query=query, size=args.max_events,
                     sort=[{"@timestamp": "asc"}])
    events = [hit["_source"] for hit in resp.get("hits", {}).get("hits", [])]
    return events, es


def fetch_from_file(path: Path) -> tuple[list[dict], None]:
    """Load feedback events from an offline JSON dump (list of documents)."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "hits" in data:           # raw ES response
        data = [h["_source"] for h in data["hits"]["hits"]]
    if not isinstance(data, list):
        raise SystemExit("ERROR: --from-file must contain a JSON list of events")
    return data, None


def resolve_car(es, cars_index: str, car_id: str) -> dict | None:
    """Look up a car document so we can attach its stored image (s3_key)."""
    if es is None:
        return None
    try:
        doc = es.get(index=cars_index, id=car_id)
        return doc.get("_source")
    except Exception:  # noqa: BLE001  (NotFoundError and friends)
        return None


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def build_records(events: list[dict], cfg: dict, es, cars_index: str
                  ) -> tuple[list[dict], list[dict], int]:
    """Split feedback events into labelled training records and review records.

    Returns (labelled, review, ignored) where ``ignored`` counts well-formed
    events that are simply not relevant to this model (e.g. a plain ``confirm``
    for the OCR model adds no error signal) plus malformed events.
    """
    label_actions = set(cfg["label_actions"])
    review_actions = set(cfg["review_actions"])
    labelled: list[dict] = []
    review: list[dict] = []
    ignored = 0

    for ev in events:
        action = ev.get("action")
        disputed = ev.get("disputed", False)
        result_id = ev.get("result_id")
        if not action or not result_id:
            ignored += 1
            continue

        # "disputed" is a flag that can ride along with any action.
        effective = "disputed" if disputed else action
        car = resolve_car(es, cars_index, result_id)
        base = {
            "result_id": result_id,
            "action": action,
            "disputed": disputed,
            "comment": ev.get("comment"),
            "feedback_ts": ev.get("@timestamp"),
            "s3_key": (car or {}).get("s3_key"),
            "indexed_plate": (car or {}).get("plate_number"),
        }

        if effective in label_actions or action in label_actions:
            rec = dict(base)
            # For OCR the corrected plate is the ground-truth label; for a plain
            # "confirm" the already-indexed plate is the confirmed label.
            rec["label_plate"] = ev.get("corrected_plate") or base["indexed_plate"]
            if rec["label_plate"]:
                labelled.append(rec)
            else:
                review.append(base)  # no usable label -> send to review
        elif effective in review_actions or action in review_actions:
            review.append(base)
        else:
            # well-formed but not relevant to this model
            ignored += 1

    return labelled, review, ignored


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export operator feedback into training data")
    parser.add_argument("--model", choices=sorted(WORKSPACES), required=True)
    parser.add_argument("--since-days", type=int, default=30,
                        help="time window for feedback events (default: 30)")
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--from-file", type=Path,
                        help="read events from a JSON dump instead of Elasticsearch")
    parser.add_argument("--dry-run", action="store_true",
                        help="report counts but do not write any files")
    # Elasticsearch connection (defaults mirror the app's env vars)
    parser.add_argument("--es-host", default=os.getenv("ELASTICSEARCH_HOST", "localhost"))
    parser.add_argument("--es-port", type=int,
                        default=int(os.getenv("ELASTICSEARCH_PORT", "9200")))
    parser.add_argument("--es-user", default=os.getenv("ELASTICSEARCH_USERNAME", ""))
    parser.add_argument("--es-password", default=os.getenv("ELASTICSEARCH_PASSWORD", ""))
    parser.add_argument("--es-ssl", action="store_true")
    parser.add_argument("--cars-index", default=os.getenv("ELASTICSEARCH_INDEX", "cars"))
    args = parser.parse_args()

    cfg = WORKSPACES[args.model]

    if args.from_file:
        events, es = fetch_from_file(args.from_file)
        print(f"Loaded {len(events)} feedback events from {args.from_file}")
    else:
        events, es = fetch_from_es(args)
        print(f"Fetched {len(events)} feedback events from "
              f"'{FEEDBACK_INDEX}' (last {args.since_days} days)")

    labelled, review, ignored = build_records(events, cfg, es, args.cars_index)

    print(f"Model '{args.model}':")
    print(f"  labelled training records : {len(labelled)}")
    print(f"  review / hard-case records: {len(review)}")
    print(f"  ignored (not relevant)    : {ignored}")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return 0

    if not labelled and not review:
        print("\nNothing to export — no relevant feedback in the window.")
        return 0

    out_dir = REPO_ROOT / str(cfg["dir"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    written: list[str] = []

    if labelled:
        for name in (f"labelled_{stamp}.jsonl", "latest.jsonl"):
            p = out_dir / name
            write_jsonl(p, labelled)
            written.append(str(p.relative_to(REPO_ROOT)))
    if review:
        p = out_dir / f"review_{stamp}.jsonl"
        write_jsonl(p, review)
        written.append(str(p.relative_to(REPO_ROOT)))

    manifest = {
        "model": args.model,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "window_days": args.since_days,
        "source": str(args.from_file) if args.from_file else FEEDBACK_INDEX,
        "labelled": len(labelled),
        "review": len(review),
    }
    mpath = out_dir / "export_manifest.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    written.append(str(mpath.relative_to(REPO_ROOT)))

    print("\nWritten:")
    for w in written:
        print(f"  {w}")
    print("\nThe ml/<workspace> prepare step will pick up latest.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
