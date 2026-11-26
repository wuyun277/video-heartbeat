from pathlib import Path
import json
import cv2
from scipy.interpolate import PchipInterpolator

root = Path(__file__).resolve().parents[1]
source = root / 'input/ai_face_source.png'
destination = root / 'input/ai_face_30s.mp4'
if destination.exists():
    raise FileExistsError(destination)
portrait = cv2.imread(str(source))
if portrait is None:
    raise ValueError('Cannot read portrait')
width, height, fps, frames = 1280, 720, 30, 900
portrait = cv2.resize(portrait, (width, height), interpolation=cv2.INTER_AREA)
# Slow, non-repeating geometric motion. No pulse/color modulation.
times = [0, 4.3, 9.1, 14.8, 20.2, 25.7, 30]
dx = PchipInterpolator(times, [0, 2.5, -1.2, 1.7, -2.2, 0.5, 0])
dy = PchipInterpolator(times, [0, -1.1, 0.6, -0.4, 1.3, -0.7, 0])
angle = PchipInterpolator(times, [0, 0.18, -0.12, 0.09, -0.15, 0.07, 0])
zoom = PchipInterpolator(times, [1.012, 1.015, 1.011, 1.016, 1.013, 1.017, 1.012])
writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
if not writer.isOpened():
    raise RuntimeError('MP4 writer failed')
try:
    for index in range(frames):
        t = index / fps
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), float(angle(t)), float(zoom(t)))
        matrix[:, 2] += [float(dx(t)), float(dy(t))]
        frame = cv2.warpAffine(portrait, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        writer.write(frame)
finally:
    writer.release()
cap = cv2.VideoCapture(str(destination))
decoded = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    decoded += 1
actual_fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
assert decoded == frames and abs(actual_fps - fps) < 0.01
print(json.dumps({'path':str(destination), 'frames':decoded, 'fps':actual_fps, 'duration_s':decoded/actual_fps, 'size':[width,height], 'bytes':destination.stat().st_size}))
