"""Конвертация экспорта Label Studio (JSON-MIN) детекции в YOLO-формат.

Использование:
    python ls_to_yolo.py export_detection.json out_labels/

На выходе для каждого изображения создаётся .txt с строками:
    <class> <x_center> <y_center> <width> <height>   (нормализованные).
Класс всегда 0 (plate). Изображения с флагом no_plate пропускаются.
"""

import json, sys, os


def convert(export_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(export_path, encoding="utf-8") as f:
        tasks = json.load(f)

    written, skipped = 0, 0
    for t in tasks:
        if "no_plate" in t.get("flags", []):
            skipped += 1
            continue
        stem = os.path.splitext(os.path.basename(t["image"]))[0]
        lines = []
        for b in t.get("bbox", []):
            # Label Studio JSON-MIN хранит x,y,width,height в ПРОЦЕНТАХ от размера
            x = b["x"] / 100.0
            y = b["y"] / 100.0
            w = b["width"] / 100.0
            h = b["height"] / 100.0
            xc, yc = x + w / 2.0, y + h / 2.0  # top-left -> center
            lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        with open(os.path.join(out_dir, stem + ".txt"), "w") as o:
            o.write("\n".join(lines))
        written += 1
    print(f"YOLO labels written: {written}, skipped (no_plate): {skipped}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python ls_to_yolo.py <export.json> <out_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
