"""Авто-проверки разметки перед review (acceptance gate).

Проверяет:
  - детекция: bbox в пределах [0,100]% , положительная площадь, наличие класса;
  - OCR: строка матчит регекс российского номера, символы из алфавита;
  - общий: отсутствие пустых задач.

Выход: код 0 если ошибок нет, иначе печатает список нарушений и код 1.
Использование:
    python validate_annotations.py --detection export_detection.json
    python validate_annotations.py --ocr export_ocr.json
"""

import json, re, sys, argparse

PLATE_RE = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")
ALPHABET = set("1234567890ABEKMHOPCTYX")


def check_detection(path):
    errs = []
    tasks = json.load(open(path, encoding="utf-8"))
    for t in tasks:
        tid = t.get("id")
        if "no_plate" in t.get("flags", []):
            continue
        boxes = t.get("bbox", [])
        if not boxes:
            errs.append(f"[det:{tid}] нет bbox и не помечено no_plate")
        for b in boxes:
            if not (0 <= b["x"] <= 100 and 0 <= b["y"] <= 100):
                errs.append(f"[det:{tid}] координаты вне диапазона")
            if b["width"] <= 0 or b["height"] <= 0:
                errs.append(f"[det:{tid}] неположительная площадь bbox")
            if b["x"] + b["width"] > 100.5 or b["y"] + b["height"] > 100.5:
                errs.append(f"[det:{tid}] bbox выходит за границы изображения")
            if "plate" not in b.get("rectanglelabels", []):
                errs.append(f"[det:{tid}] отсутствует класс plate")
    return errs


def check_ocr(path):
    errs = []
    tasks = json.load(open(path, encoding="utf-8"))
    for t in tasks:
        tid = t.get("id")
        s = (t.get("transcription") or "").strip()
        if not s:
            errs.append(f"[ocr:{tid}] пустая транскрипция")
            continue
        if set(s) - ALPHABET:
            errs.append(f"[ocr:{tid}] символы вне алфавита: {s}")
        if not PLATE_RE.match(s):
            errs.append(f"[ocr:{tid}] не соответствует формату номера: {s}")
    return errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection")
    ap.add_argument("--ocr")
    a = ap.parse_args()
    errs = []
    if a.detection:
        errs += check_detection(a.detection)
    if a.ocr:
        errs += check_ocr(a.ocr)
    if errs:
        print("FAILED:", len(errs), "нарушений")
        for e in errs:
            print(" -", e)
        sys.exit(1)
    print("OK: все авто-проверки пройдены")
