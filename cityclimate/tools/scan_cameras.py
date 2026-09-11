"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: tools/scan_cameras.py — Scan all camera indices 0-9 and save a
JPG for each one found, to help identify which cv2.VideoCapture device
index corresponds to the physical camera intended for the board.
"""
import cv2
from pathlib import Path

out = Path(".")
print("Suche Kameras (Index 0-9) ...\n")
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        continue
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"Index {i}: gefunden aber kein Frame")
        continue
    path = out / f"cam_{i}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"Index {i}: {frame.shape[1]}x{frame.shape[0]}px  -> {path}")
print("\nFertig. Schau welches Bild farbig und hell ist -> das ist der richtige Index.")
