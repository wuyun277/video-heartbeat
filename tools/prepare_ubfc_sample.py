"""Extract a short UBFC-rPPG Dataset 1 clip and its matching sensor samples.

This is a data-preparation utility; it does not run the heartbeat estimator.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


# 截取 UBFC Dataset 1 视频开头的指定时长，同步导出对应传感器记录。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Official Dataset 1 vid.avi")
    parser.add_argument("ground_truth", type=Path, help="Matching gtdump.xmp")
    parser.add_argument("output_video", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.output_video.exists() or args.output_csv.exists():
        parser.error("Refusing to overwrite an existing output")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not 1 <= fps <= 240 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Invalid source video FPS or dimensions")

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create video: {args.output_video}")

    # 按原始帧顺序和名义帧率截取视频，此工具仅负责准备数据。
    target_frames = round(args.seconds * fps)
    written = 0
    try:
        while written < target_frames:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written == 0:
        args.output_video.unlink(missing_ok=True)
        raise RuntimeError("No frames could be decoded")

    # 传感器时间戳以录制起点为零，单位为毫秒，导出时转换为秒。
    duration_s = written / fps
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.ground_truth.open(encoding="utf-8") as source, args.output_csv.open(
        "x", newline="", encoding="utf-8"
    ) as target:
        writer_csv = csv.writer(target)
        writer_csv.writerow(("time_s", "heart_rate_bpm", "spo2_percent", "ppg"))
        for line in source:
            fields = line.strip().split(",")
            if len(fields) != 4:
                continue
            time_s = float(fields[0]) / 1000.0
            if time_s >= duration_s:
                break
            writer_csv.writerow((f"{time_s:.3f}", *fields[1:]))
            kept += 1
    print(f"Wrote {written} frames ({duration_s:.3f} s) and {kept} reference samples")


if __name__ == "__main__":
    main()
