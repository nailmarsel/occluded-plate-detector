#!/usr/bin/env python3
"""Build the old-vs-fine-tuned comparison table for the plate models.

Reads the JSON files written by
    ml/plate_detector/scripts/compare_plate_detectors.py
    ml/plate_ocr/scripts/compare_plate_ocr.py
and writes
    reports/models/FINETUNE_COMPARISON.md   (table for the report)
    reports/models/finetune/comparison.csv  (same data for Excel)
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KEY_METRICS = {"mAP50", "mAP50-95", "recall", "exact_match", "cer"}
DEFAULT_INPUTS = [
    REPO_ROOT / "reports" / "models" / "finetune" / "plate_detector.json",
    REPO_ROOT / "reports" / "models" / "finetune" / "plate_ocr.json",
]


def fmt(metric: str, value: float) -> str:
    if metric == "ms_per_image":
        return f"{value:.1f}"
    return f"{value:.4f}"


def delta(row: dict) -> tuple[str, str]:
    diff = row["candidate"] - row["baseline"]
    if row["metric"] == "ms_per_image":
        text = f"{diff:+.1f}"
    else:
        text = f"{diff:+.4f}"
        if row["baseline"]:
            text += f" ({diff / row['baseline'] * 100:+.1f}%)"
    eps = 0.05 if row["metric"] == "ms_per_image" else 5e-4
    if abs(diff) < eps:
        verdict = "≈"
    else:
        better = diff > 0 if row["higher_is_better"] else diff < 0
        verdict = "лучше" if better else "хуже"
    return text, verdict


def rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path


def describe_finetune(data: dict) -> list[str]:
    ft = data.get("finetune") or {}
    if not ft:
        return []
    parts = [f"{ft.get('train_minutes', '?')} мин", f"устройство {ft.get('device', '?')}"]
    if data["model"] == "license_plate_detector":
        parts += [
            f"lr0={ft.get('lr0')}",
            f"imgsz={ft.get('imgsz')}",
            f"доля перекрытых копий в train={ft.get('occlusion_fraction')}",
        ]
    else:
        parts += [
            f"lr={ft.get('lr')}",
            f"лучшая эпоха {ft.get('best_epoch')}",
            f"val exact: {ft.get('base_val_exact')} → {ft.get('best_val_exact')}",
        ]
    return [f"- Дообучение: {', '.join(str(p) for p in parts)}."]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--markdown", type=Path, default=REPO_ROOT / "reports" / "models" / "FINETUNE_COMPARISON.md"
    )
    parser.add_argument(
        "--csv", type=Path, default=REPO_ROOT / "reports" / "models" / "finetune" / "comparison.csv"
    )
    args = parser.parse_args()

    reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.inputs if p.exists()]
    missing = [str(p) for p in args.inputs if not p.exists()]
    if not reports:
        raise SystemExit("No comparison results found. Run the compare scripts first.")

    header = [
        "| Модель | Тестовый набор | N | Метрика | Исходная | Дообученная | Δ | Итог |",
        "|---|---|---:|---|---:|---:|---:|:---:|",
    ]
    lines = [
        "# Сравнение исходных и дообученных моделей",
        "",
        f"Сформировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
        "Обе модели в каждой паре оценены на одних и тех же изображениях из test-выборки, "
        "которая не участвовала ни в исходном обучении, ни в дообучении.",
        "",
        "## Ключевые метрики",
        "",
        *header,
    ]
    detail = ["", "## Все метрики", "", *header]
    csv_rows = []
    for data in reports:
        for row in data["rows"]:
            d_text, verdict = delta(row)
            line = (
                f"| {data['title']} | {row['dataset_title']} | {row['n']} | {row['metric_title']} | "
                f"{fmt(row['metric'], row['baseline'])} | {fmt(row['metric'], row['candidate'])} | "
                f"{d_text} | {verdict} |"
            )
            detail.append(line)
            if row["metric"] in KEY_METRICS:
                lines.append(line)
            csv_rows.append(
                {
                    "model": data["model"],
                    "dataset": row["dataset"],
                    "n": row["n"],
                    "metric": row["metric"],
                    "baseline": round(row["baseline"], 6),
                    "finetuned": round(row["candidate"], 6),
                    "delta": round(row["candidate"] - row["baseline"], 6),
                    "verdict": verdict,
                }
            )

    lines += detail
    lines += ["", "## Условия", ""]
    for data in reports:
        lines += [
            f"**{data['title']}**",
            "",
            f"- Исходная модель: `{rel(data['baseline'])}`.",
            f"- Дообученная модель: `{rel(data['candidate'])}`.",
            *describe_finetune(data),
            f"- Оценка: устройство {data.get('device')}, {data['evaluated_at']}.",
            "",
        ]
    lines += [
        "## Как читать",
        "",
        "- **Детектор**: mAP@0.5 — основная метрика (цель проекта ≥ 0.90). Набор «с перекрытием» — те же "
        "тестовые кадры, где часть каждого номера закрыта прямоугольником (детерминированно, seed фиксирован).",
        "- **OCR**: exact match — доля номеров, распознанных целиком без ошибок (цель проекта ≥ 0.95); "
        "CER — доля ошибочных символов. «Ухудшенные кропы» — те же номера с фиксированным размытием, шумом, "
        "пониженным контрастом и наклоном.",
        "- Аугментации при дообучении похожи на искажения «сложных» тестовых наборов (перекрытия для детектора, "
        "размытие/контраст для OCR), поэтому прирост там ожидаемо больше. Прирост на исходных изображениях "
        "показывает, что модель не стала хуже на обычных данных.",
        "- Скорость инференса зависит от устройства и загрузки машины; сравнивайте только внутри одного запуска.",
    ]
    if missing:
        lines += ["", f"_Нет результатов для: {', '.join(rel(m) for m in missing)}._"]

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {args.markdown}\nSaved: {args.csv}\n")
    print("\n".join(lines[6 : lines.index("## Все метрики") - 1]))


if __name__ == "__main__":
    main()
