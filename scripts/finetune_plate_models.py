#!/usr/bin/env python3
"""One command: fine-tune plate detector + plate OCR and build the comparison table.

Steps (each can be skipped):
    1. prepare   — download HF datasets and build YOLO dataset / OCR manifests
    2. finetune  — fine-tune both models starting from src/models/*.pt
    3. compare   — evaluate old vs fine-tuned on the same test sets
    4. report    — reports/models/FINETUNE_COMPARISON.md + finetune/comparison.csv
    5. publish   — only with --publish: replace src/models/*.pt for models whose
                   main metric on the clean test did not get worse
                   (old files are backed up to src/models/rollback/)

Examples:
    python scripts/finetune_plate_models.py                       # full run
    python scripts/finetune_plate_models.py --quick               # ~10 min smoke run
    python scripts/finetune_plate_models.py --models ocr --skip-prepare
    python scripts/finetune_plate_models.py --only report
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "ml" / "plate_detector"
OCR = ROOT / "ml" / "plate_ocr"
REPORT_DIR = ROOT / "reports" / "models" / "finetune"
STEPS = ("prepare", "finetune", "compare", "report", "publish")

DET_CANDIDATE = DET / "runs" / "finetune_plate_yolo" / "weights" / "best.pt"
OCR_CANDIDATE = OCR / "runs" / "finetune_plate_ocr" / "best.pt"
ARTIFACTS = {
    "detector": (
        DET_CANDIDATE,
        ROOT / "src" / "models" / "license_plate_detector.pt",
        REPORT_DIR / "plate_detector.json",
        "mAP50",
    ),
    "ocr": (
        OCR_CANDIDATE,
        ROOT / "src" / "models" / "plate_ocr.pt",
        REPORT_DIR / "plate_ocr.json",
        "exact_match",
    ),
}


def run(cmd: list[str], title: str) -> bool:
    print(f"\n{'=' * 70}\n▶ {title}\n  {' '.join(cmd)}\n{'=' * 70}", flush=True)
    started = time.monotonic()
    code = subprocess.run(cmd, cwd=ROOT).returncode
    print(
        f"◀ {title}: {'OK' if code == 0 else f'ошибка (код {code})'} за {(time.monotonic() - started) / 60:.1f} мин",
        flush=True,
    )
    return code == 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    p.add_argument("--models", nargs="+", choices=("detector", "ocr"), default=["detector", "ocr"])
    p.add_argument("--only", nargs="+", choices=STEPS, help="Run only these steps.")
    p.add_argument("--skip-prepare", action="store_true", help="Datasets are already prepared.")
    p.add_argument(
        "--publish", action="store_true", help="Replace src/models/*.pt when the model did not get worse."
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: 1 epoch, 5 min per model, 10%% of data, 300 test samples.",
    )
    p.add_argument("--device", default="auto", help="auto | cpu | 0 (CUDA GPU) | mps (eval only)")
    p.add_argument("--max-minutes", type=float, default=25, help="Training budget per model (≤ 30).")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (optional).")
    g = p.add_argument_group("detector")
    g.add_argument("--det-epochs", type=int, default=10)
    g.add_argument("--det-imgsz", type=int, default=960)
    g.add_argument("--det-batch", type=int, default=4)
    g.add_argument("--det-fraction", type=float, default=1.0)
    g.add_argument("--occlusion-fraction", type=float, default=0.3)
    g = p.add_argument_group("ocr")
    g.add_argument("--ocr-epochs", type=int, default=8)
    g.add_argument("--ocr-batch", type=int, default=64)
    g = p.add_argument_group("comparison")
    g.add_argument("--limit", type=int, default=0, help="Test samples per model (0 = whole test split).")
    return p


def publish(models: list[str]) -> None:
    for name in models:
        candidate, artifact, report, metric = ARTIFACTS[name]
        if not (candidate.exists() and report.exists()):
            print(f"[publish] {name}: нет дообученной модели или результатов сравнения — пропуск")
            continue
        rows = json.loads(report.read_text(encoding="utf-8"))["rows"]
        row = next(r for r in rows if r["dataset"] == "clean" and r["metric"] == metric)
        if row["candidate"] < row["baseline"]:
            print(
                f"[publish] {name}: {metric} на чистом тесте хуже "
                f"({row['candidate']:.4f} < {row['baseline']:.4f}) — не публикую"
            )
            continue
        rollback = artifact.parent / "rollback" / artifact.name
        rollback.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, rollback)
        shutil.copy2(candidate, artifact)
        print(
            f"[publish] {name}: {artifact.relative_to(ROOT)} обновлён, старая версия в {rollback.relative_to(ROOT)}"
        )


def main() -> int:
    args = build_parser().parse_args()
    py = sys.executable
    if args.quick:
        args.det_epochs = args.ocr_epochs = 1
        args.max_minutes = min(args.max_minutes, 5)
        args.det_fraction = min(args.det_fraction, 0.1)
        args.det_imgsz = min(args.det_imgsz, 640)
        args.limit = args.limit or 300
    steps = set(args.only or STEPS)
    if args.skip_prepare:
        steps.discard("prepare")
    if not args.publish:
        steps.discard("publish")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    device = ["--device", args.device]
    budget = ["--max-minutes", str(args.max_minutes)]
    limit = ["--limit", str(args.limit)] if args.limit else []
    ok = True

    if "prepare" in steps:
        if "detector" in args.models:
            ok &= run(
                [py, str(DET / "scripts" / "train_plate_detector.py"), "--prepare-only"],
                "Подготовка датасета детекции (Hugging Face → YOLO)",
            )
        if "ocr" in args.models:
            ok &= run(
                [py, str(OCR / "scripts" / "prepare_ocr_dataset.py")],
                "Подготовка датасета OCR (Hugging Face → манифесты)",
            )
        if not ok:
            print("\nПодготовка данных не удалась — дальше не иду.")
            return 1

    if "finetune" in steps:
        if "detector" in args.models:
            ok &= run(
                [
                    py,
                    str(DET / "scripts" / "finetune_plate_detector.py"),
                    *device,
                    *budget,
                    "--epochs",
                    str(args.det_epochs),
                    "--imgsz",
                    str(args.det_imgsz),
                    "--batch",
                    str(args.det_batch),
                    "--fraction",
                    str(args.det_fraction),
                    "--occlusion-fraction",
                    str(args.occlusion_fraction),
                ],
                "Дообучение детектора номеров",
            )
        if "ocr" in args.models:
            ok &= run(
                [
                    py,
                    str(OCR / "scripts" / "finetune_plate_ocr.py"),
                    *device,
                    *budget,
                    "--epochs",
                    str(args.ocr_epochs),
                    "--batch-size",
                    str(args.ocr_batch),
                ],
                "Дообучение OCR",
            )

    if "compare" in steps:
        if "detector" in args.models:
            if DET_CANDIDATE.exists():
                ok &= run(
                    [
                        py,
                        str(DET / "scripts" / "compare_plate_detectors.py"),
                        *device,
                        *limit,
                        "--imgsz",
                        str(args.det_imgsz),
                    ],
                    "Сравнение детекторов",
                )
            else:
                print(f"\n[compare] Нет дообученного детектора: {DET_CANDIDATE}")
        if "ocr" in args.models:
            if OCR_CANDIDATE.exists():
                ok &= run(
                    [
                        py,
                        str(OCR / "scripts" / "compare_plate_ocr.py"),
                        *device,
                        *limit,
                        "--batch-size",
                        str(args.ocr_batch),
                    ],
                    "Сравнение OCR",
                )
            else:
                print(
                    f"\n[compare] Нет дообученного OCR: {OCR_CANDIDATE} "
                    "(если дообучение прошло, значит модель не обогнала исходную на val)"
                )

    if "report" in steps:
        # Always both models: results of a model that was not re-run this time stay in the table.
        ok &= run([py, str(ROOT / "scripts" / "build_finetune_report.py")], "Сборка таблицы сравнения")

    if "publish" in steps:
        publish(args.models)

    print("\nГотово." if ok else "\nЗавершено с ошибками — см. лог выше.")
    if (ROOT / "reports" / "models" / "FINETUNE_COMPARISON.md").exists():
        print("Таблица: reports/models/FINETUNE_COMPARISON.md")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
