"""
CityClimate Board — Research Seminar, Summer Term 2026
TH Köln

Team: Alessio Fiorito, Bakir Ahmetbegovic, Segmen Bagcivan

Module: tools/realsense_test.py — Quick test: grab one RGB frame via
pyrealsense2 and save it as realsense_rgb.jpg, to verify the RealSense
camera and pyrealsense2 installation are working correctly.
"""
try:
    import pyrealsense2 as rs
except ImportError:
    print("FEHLER: pyrealsense2 nicht installiert.")
    print("Bitte ausfuehren: pip install pyrealsense2")
    raise SystemExit(1)

import numpy as np
import cv2

pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starte RealSense RGB-Stream...")
try:
    pipeline.start(cfg)
except Exception as e:
    print(f"FEHLER beim Starten: {e}")
    print("Ist die RealSense per USB3 angeschlossen?")
    raise SystemExit(1)

try:
    for _ in range(10):   # wait a few frames until auto-exposure stabilizes
        pipeline.wait_for_frames()
    frames      = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Kein Color-Frame erhalten.")
        raise SystemExit(1)
    img = np.asanyarray(color_frame.get_data())
    cv2.imwrite("realsense_rgb.jpg", img)
    print(f"Gespeichert: realsense_rgb.jpg  ({img.shape[1]}x{img.shape[0]}px)")
finally:
    pipeline.stop()
