"""OpenCV dashboard and bounded history of the displayed heart rate."""
from collections import deque
import math

import cv2
import numpy as np

SIZE = (1440, 900)
BG, PANEL, INK, MUTED = (20, 19, 16), (34, 32, 27), (244, 242, 232), (168, 164, 150)
GREEN, AMBER, GRID = (161, 222, 95), (95, 190, 249), (65, 61, 50)


# 保存有限时长的心率趋势，以 None 标记无效读数形成的断点。
class HeartRateTrend:
    def __init__(self, seconds=60):
        self.seconds = seconds
        self.points = deque()
        self.last_estimate = None
        self.was_valid = False

    def record(self, timestamp, bpm, estimate_time):
        # 只保存估计更新及有效性变化；None 作为断点，不能用 0 冒充无效心率。
        valid = bpm is not None
        if not self.points or valid != self.was_valid or estimate_time != self.last_estimate:
            self.points.append((timestamp, bpm))
        self.was_valid = valid
        self.last_estimate = estimate_time
        while self.points and self.points[0][0] < timestamp - self.seconds:
            self.points.popleft()

    def clear(self):
        # 换人或按 R 时同时清除曲线及状态，避免新旧受试者的读数连在一起。
        self.points.clear()
        self.last_estimate = None
        self.was_valid = False


# 统一仪表盘文字的字体、颜色和抗锯齿设置。
def text(image, message, x, y, size=.6, color=INK, thickness=1):
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)


# 绘制实心面板背景，供各显示区块复用。
def panel(canvas, x, y, w, h):
    cv2.rectangle(canvas, (x, y), (x+w, y+h), PANEL, -1)


# 按源时间绘制趋势，并根据可见读数调整心率纵轴。
def draw_trend(canvas, trend, timestamp, bpm, args):
    # 横轴使用源时间，不是处理耗时；较慢的离线处理也应保持正确的视频时间轴。
    panel(canvas, 482, 372, 934, 464)
    text(canvas, "HEART RATE OVER TIME", 510, 410, .72, INK, 2)
    text(canvas, "Displayed bpm | gaps = unavailable | source time in seconds", 510, 440, .52, MUTED)
    x, y, w, h = 553, 493, 827, 258
    end = max(trend.seconds, timestamp)
    start = end - trend.seconds
    points = list(trend.points)
    if bpm is not None:
        points.append((timestamp, bpm))
    values = [value for t, value in points if value is not None and start <= t <= end]
    low, high = args.min_bpm, args.max_bpm
    if values:
        low = max(args.min_bpm, math.floor((min(values)-5)/10)*10)
        high = min(args.max_bpm, math.ceil((max(values)+5)/10)*10)
        if high-low < 20:
            low = max(args.min_bpm, high-20)
            high = min(args.max_bpm, low+20)
    for value in np.linspace(low, high, 5):
        py = int(y+h-(value-low)/(high-low)*h)
        cv2.line(canvas, (x, py), (x+w, py), GRID, 1)
        text(canvas, f"{value:.0f}", x-47, py+5, .48, MUTED)
    for t in np.linspace(start, end, 7):
        px = int(x+(t-start)/trend.seconds*w)
        cv2.line(canvas, (px, y), (px, y+h), GRID, 1)
        text(canvas, f"{t:.0f}s", px-14, y+h+28, .46, MUTED)
    text(canvas, "bpm", x-47, y-15, .46, MUTED)
    previous = None
    for t, value in points:
        if value is None or not start <= t <= end:
            # 无效片段必须断开，不能把两侧有效点跨越缺失时段连起来。
            previous = None
            continue
        p = (int(x+(t-start)/trend.seconds*w), int(y+h-(value-low)/(high-low)*h))
        if previous is not None:
            cv2.line(canvas, previous, p, GREEN, 2, cv2.LINE_AA)
        cv2.circle(canvas, p, 3, GREEN, -1, cv2.LINE_AA)
        previous = p
    now_x = int(x+(timestamp-start)/trend.seconds*w)
    cv2.line(canvas, (now_x, y), (now_x, y+h), MUTED, 1)
    if not values:
        text(canvas, "No accepted readings in this time range", x+110, y+116, .7, INK)
        text(canvas, "Keep the face visible and follow the status guidance above.", x+80, y+151, .48, MUTED)
    text(canvas, "Scale adapts to visible values; each reading summarizes a trailing window.", 510, 812, .48, MUTED)


# 组合视频预览、当前心率、信号状态和趋势，返回 BGR 画布。
def render(frame, result, bpm, status, progress, args, trend, timestamp):
    canvas = np.full((SIZE[1], SIZE[0], 3), BG, dtype=np.uint8)
    text(canvas, "PULSE / VIDEO", 28, 44, .9, INK, 2)
    text(canvas, f"SOURCE TIME  {timestamp:06.1f}s", 1100, 42, .65, MUTED)
    panel(canvas, 24, 72, 434, 764)
    text(canvas, "VIDEO & SAMPLING REGIONS", 44, 106, .56, MUTED)
    factor = min(410/frame.shape[1], 666/frame.shape[0])
    preview = cv2.resize(frame, (max(1, round(frame.shape[1]*factor)), max(1, round(frame.shape[0]*factor))))
    px = 36+(410-preview.shape[1])//2
    py = 125+(666-preview.shape[0])//2
    canvas[py:py+preview.shape[0], px:px+preview.shape[1]] = preview
    method = getattr(args, "method", "pos")
    sampling_label = "EfficientPhys enlarged face crop" if method == "efficientphys" else "Forehead + left cheek + right cheek"
    text(canvas, sampling_label, 44, 817, .5, MUTED)

    panel(canvas, 482, 72, 934, 280)
    color = GREEN if bpm is not None else AMBER
    text(canvas, "CURRENT HEART RATE", 510, 113, .65, MUTED)
    text(canvas, f"{bpm:.0f}" if bpm is not None else "--", 508, 223, 2.9, color, 4)
    text(canvas, "bpm", 720, 216, .8, MUTED)
    text(canvas, "Accepted by signal checks" if bpm is not None else "No accepted reading", 510, 267, .54, color)
    if bpm is None and result and result.get('bpm') is not None:
        # 黄色小字是最近窗口候选，与主数字和正式趋势严格分开。
        text(canvas, f"Last window candidate: {result['bpm']:.0f} bpm", 510, 292, .46, AMBER)
    text(canvas, f"Window {args.window:g}s  /  Update {args.update:g}s", 510, 317, .5, MUTED)

    hints = {
        "GOOD": ("SIGNAL ACCEPTED", "Reading summarizes the latest analysis window."),
        "HOLD STILL": ("TOO MUCH MOVEMENT", "Keep your head and camera steady."),
        "WEAK SIGNAL": ("WEAK PULSE SIGNAL", "Expose forehead and cheeks; use even light."),
        "SINGLE ROI": ("ONE REGION ONLY", "Candidate only; a second region must agree."),
        "AMBIGUOUS": ("COMPETING FREQUENCIES", "More than one frequency has similar support."),
        "ROI QUALITY": ("REGION QUALITY CHANGED", "Too few supporting regions are usable now."),
        "LOW LIGHT": ("LIGHT TOO DIM", "Increase steady, even lighting."),
        "OVEREXPOSED": ("LIGHT TOO BRIGHT", "Reduce glare and direct light on the face."),
        "NO FACE": ("FACE NOT FOUND", "Face the camera and keep your face visible."),
        "FACE / LIGHT INVALID": ("CHECK FACE / LIGHT", "Keep sampling regions visible and well lit."),
        "COLLECTING": ("COLLECTING SIGNAL", "Waiting for one complete analysis window."),
        "TIMING / FPS INVALID": ("CHECK FRAME TIMING", "Sampling is too slow or has large gaps."),
    }
    title, hint = hints.get(status, (status, "Waiting for the next estimate."))
    text(canvas, title, 844, 114, .72, color, 2)
    text(canvas, hint, 844, 151, .5, MUTED)
    text(canvas, f"BUFFER  {progress:.1f} / {args.window:g} s", 844, 195, .58, INK)
    cv2.rectangle(canvas, (844, 211), (1385, 221), GRID, -1)
    cv2.rectangle(canvas, (844, 211), (844+int(541*min(1, max(0, progress/args.window))), 221), color, -1)
    text(canvas, f"Recent window score: {result['quality']:.2f}" if result else "Recent window score: --", 844, 264, .54, MUTED)
    labels = ["F", "L", "R"]
    support = "MODEL" if method == "efficientphys" and result else (
        "/".join(labels[i] for i in result.get('support', [])) if result else "")
    text(canvas, f"Window support: {support or '--'} | score is not accuracy", 844, 291, .48, MUTED)
    text(canvas, f"Source sampling: {result['fs']:.1f} Hz" if result else "Source sampling: waiting for full window", 844, 321, .5, MUTED)
    draw_trend(canvas, trend, timestamp, bpm, args)
    text(canvas, "Q / ESC  Exit     R  Restart & clear trend", 28, 877, .53, MUTED)
    text(canvas, "Video-based heart rate estimate | Research demo", 932, 877, .5, MUTED)
    return canvas
