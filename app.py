"""Single-person POS heart-rate demo. See README.md before use."""
from __future__ import annotations

import argparse
import csv
import logging
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy import signal

from settings import DEFAULT_CONFIG, directory, file_path, load_config
from dashboard import SIZE, HeartRateTrend, render
from fusion import fuse_spectra, options as fusion_options
from efficientphys import EfficientPhysEstimator, face_box, face_crop

LOGGER = logging.getLogger("video_heartbeat")


# 根据 MediaPipe 人脸关键点定义采样区，避开眼睛和嘴部。
REGIONS = {
    "forehead": [109, 10, 338, 337, 151, 108],
    "left cheek": [50, 101, 205, 207, 187, 123],
    "right cheek": [280, 330, 425, 427, 411, 352],
}
COLORS = [(80, 220, 100), (230, 190, 70), (180, 110, 240)]


# 从 RGB 均值序列提取 POS 脉搏波形，fs 表示实际采样率（Hz）。
def pos(rgb: np.ndarray, fs: float) -> np.ndarray:
    """RGB [time, 3] -> centered POS trace, normalized overlap-add."""
    # POS 内部窗口用于局部颜色归一化，与外层 15 秒心率分析窗口不同。
    length = max(8, int(round(1.6 * fs)))
    total = np.zeros(len(rgb), dtype=np.float64)
    weight = np.zeros(len(rgb), dtype=np.float64)
    for end in range(length, len(rgb) + 1):
        start = end - length
        chunk = rgb[start:end]
        # 分别消除三个颜色通道的平均亮度，再投影提取脉搏相关分量。
        normalized = chunk / np.maximum(chunk.mean(axis=0), 1e-8)
        r, g, b = normalized.T
        s1, s2 = g - b, g + b - 2 * r
        spread = np.std(s2)
        if spread < 1e-10:
            continue
        pulse = s1 + np.std(s1) / spread * s2
        total[start:end] += pulse - pulse.mean()
        weight[start:end] += 1
    return total / np.maximum(weight, 1)


# 采集额头和双颊的颜色均值、过暗及过曝比例，供质量筛选使用。
def extract_regions(frame: np.ndarray, points: np.ndarray):
    """Return per-region RGB means and exposure measures from raw pixels."""
    # 列表顺序始终与 REGIONS 一致；NaN 表示该区域本帧没有有效采样。
    # 只丢弃局部区域，避免一处被遮挡时连带停止其余两个区域的采集。
    means, dark, clipped, polygons = [], [], [], []
    for indices in REGIONS.values():
        polygon = np.round(points[indices]).astype(np.int32)
        polygons.append(polygon)
        inside = np.all((polygon[:, 0] >= 0) & (polygon[:, 0] < frame.shape[1]) &
                        (polygon[:, 1] >= 0) & (polygon[:, 1] < frame.shape[0]))
        if not inside:
            means.append([np.nan]*3)
            dark.append(0.0)
            clipped.append(0.0)
            continue
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(polygon), 255)
        pixels = frame[mask > 0]
        if len(pixels) < 100:
            means.append([np.nan]*3)
            dark.append(1.0)
            clipped.append(0.0)
            continue
        dark.append(float(np.mean(pixels.max(axis=1) < 30)))
        clipped.append(float(np.mean(pixels.max(axis=1) > 245)))
        usable = pixels[(pixels.max(axis=1) >= 30) & (pixels.max(axis=1) <= 245)]
        if len(usable) < 100:
            means.append([np.nan]*3)
            continue
        # 将 BGR 通道顺序改为 RGB，采样像素不做额外颜色变换。
        means.append(usable[:, ::-1].mean(axis=0))
    if not np.any(np.all(np.isfinite(means), axis=1)):
        return None
    return np.asarray(means), np.asarray(dark), np.asarray(clipped), polygons


# 检查采样时间轴，分别计算区域频谱，再融合一致的心率候选。
def estimate(samples, low: float, high: float, motion_limit: float, fusion=None, previous_bpm=None):
    # 每条样本：(时间戳, 3×3 RGB 均值, 三区域运动, 三区域过暗比例, 三区域过曝比例)。
    cfg = fusion_options(fusion)
    timestamps = np.asarray([s[0] for s in samples])
    intervals = np.diff(timestamps)
    median_dt = float(np.median(intervals))
    if median_dt <= 0:
        return None
    fs = 1.0 / median_dt
    if fs < max(12.0, 2 * high + 2) or intervals.max() > 0.25:
        return None
    # 按实际采集时间轴重采样 RGB 均值序列，视频帧本身不做重采样。
    uniform = np.arange(timestamps[0], timestamps[-1], median_dt)
    if len(uniform) < max(8, int(round(1.6 * fs))):
        return None
    rgb = np.asarray([s[1] for s in samples])
    traces, spectra, scores, rates = [], [], [], []
    sos = signal.butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
    # Match scipy.sosfiltfilt default padding; very short windows are unavailable.
    padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    trim = max(1, int(0.8 * fs))
    if len(uniform) <= padlen or len(uniform) - 2 * trim < 2:
        return None
    frequencies = None
    # 窗口内有缺失采样的区域不参与融合；零序列仅用于保持数组结构一致。
    usable_rgb = np.all(np.isfinite(rgb), axis=(0, 2))
    for region in range(3):
        interpolated = np.column_stack([
            np.interp(uniform, timestamps, rgb[:, region, channel]) if usable_rgb[region]
            else np.zeros(len(uniform))
            for channel in range(3)
        ])
        raw = signal.detrend(pos(interpolated, fs))
        # Retrospective trailing-window filtering, not zero-latency beat detection.
        filtered = signal.sosfiltfilt(sos, raw)
        core = filtered[trim:-trim]
        nperseg = min(len(core), int(round(10 * fs)))
        frequencies, power = signal.welch(
            core, fs=fs, nperseg=nperseg,
            nfft=max(2048, 2 ** int(np.ceil(np.log2(nperseg)))),
        )
        band = (frequencies >= low) & (frequencies <= high)
        peak_index = np.flatnonzero(band)[np.argmax(power[band])]
        peak = frequencies[peak_index]
        near = band & (np.abs(frequencies - peak) <= 0.10)
        concentration = float(power[near].sum() / max(power[band].sum(), 1e-20))
        if np.std(core) < 1e-7:
            concentration = 0.0
        traces.append(filtered)
        spectra.append(power)
        scores.append(concentration)
        rates.append(60.0 * peak)
    scores, rates = np.asarray(scores), np.asarray(rates)
    moving = np.mean(np.asarray([s[2] for s in samples]) > motion_limit, axis=0)
    dark = np.mean([s[3] for s in samples], axis=0)
    clipped = np.mean([s[4] for s in samples], axis=0)
    # 先筛选区域，再比较实际频谱峰。不能只把三个最高峰直接平均。
    eligible = usable_rgb & (scores > 0) & (moving <= cfg["max_motion_fraction"]) & (dark <= 0.4) & (clipped <= 0.3)
    fused = fuse_spectra(frequencies, spectra, eligible, low, high, cfg, previous_bpm)
    excluded = []
    for i in range(3):
        reasons = []
        if not usable_rgb[i]: reasons.append("missing skin samples")
        if scores[i] <= 0: reasons.append("flat signal")
        if moving[i] > cfg["max_motion_fraction"]: reasons.append("motion")
        if dark[i] > 0.4: reasons.append("dark")
        if clipped[i] > 0.3: reasons.append("overexposed")
        excluded.append("; ".join(reasons))
    if np.count_nonzero(dark > 0.4) >= 2:
        fused.update(status="LOW LIGHT", reason="At least two regions are too dark")
    elif np.count_nonzero(clipped > 0.3) >= 2:
        fused.update(status="OVEREXPOSED", reason="At least two regions are overexposed")
    elif np.count_nonzero(moving > cfg["max_motion_fraction"]) >= 2:
        fused.update(status="HOLD STILL", reason="At least two regions exceed the motion limit")
    selected = max(fused['support'], key=lambda i: fused['roi_scores'][i]) if fused['support'] else int(np.argmax(scores))
    rates[~usable_rgb | (scores <= 0)] = np.nan
    return dict(**fused, fs=fs, primary_scores=scores, excluded=excluded,
                trace=traces[selected], frequencies=frequencies,
                spectrum=spectra[selected], region=selected, rates=rates,
                moving=moving, dark=dark, clipped=clipped)



# 组织配置加载、视频采集、滑动窗口估计、界面显示和结果输出。
def main():
    # 先读 --config 获取默认值，再解析完整命令行，使显式参数覆盖配置。
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = config_parser.parse_known_args()
    try:
        config = load_config(config_args.config.resolve())
        fusion_cfg = fusion_options(config.get("fusion"))
        method_cfg = config.get("method", {"name": "pos"})
        if not isinstance(method_cfg, dict) or method_cfg.get("name", "pos").lower() not in ("pos", "efficientphys"):
            raise ValueError("method.name must be 'pos' or 'efficientphys'")
        method_name = method_cfg.get("name", "pos").lower()
        input_dir = directory(config["paths"]["input_dir"], config_args.config.resolve())
        output_dir = directory(config["paths"]["output_dir"], config_args.config.resolve())
    except (ValueError, TypeError) as exc:
        config_parser.error(str(exc))
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int)
    source.add_argument("--video", type=Path)
    parser.add_argument("--window", type=float, default=config["analysis"]["window"],
                        help=f"Initial collection and trailing analysis window in seconds (minimum: {config['analysis']['min_window']:g})")
    parser.add_argument("--update", type=float, default=config["analysis"]["update"],
                        help="Heart-rate estimation interval in source seconds")
    parser.add_argument("--min-bpm", type=float, default=config["analysis"]["min_bpm"])
    parser.add_argument("--max-bpm", type=float, default=config["analysis"]["max_bpm"])
    parser.add_argument("--motion-limit", type=float, default=config["analysis"]["motion_limit"],
                        help="RMS landmark motion in face-widths per second")
    parser.add_argument("--csv", type=Path, help="New CSV path; refuses to overwrite")
    parser.add_argument("--output-video", type=Path,
                        help="Save the annotated display as a new MP4 (requires --video)")
    parser.add_argument("--fps", type=float, help="Video FPS override if metadata is invalid")
    parser.add_argument("--log-interval", type=float,
                        default=config.get("logging", {}).get("interval_seconds", 1.0),
                        help="Console progress interval in wall-clock seconds (default: 1)")
    args = parser.parse_args()
    args.method = method_name
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    if not np.isfinite(args.log_interval) or args.log_interval <= 0:
        parser.error("log-interval must be finite and positive")
    if args.video is not None:
        args.video = file_path(args.video, input_dir, config_args.config.resolve())
    elif args.camera is None:
        args.video = file_path(config["paths"]["input_video"], input_dir, config_args.config.resolve())
    if args.camera is None:
        args.camera = config["camera"]["index"]
    if args.video is None and args.output_video is not None:
        parser.error("--output-video requires --video")
    args.output_video = file_path(
        args.output_video if args.output_video is not None else
        (config["paths"]["output_video"] if args.video is not None else None),
        output_dir, config_args.config.resolve(),
    )
    args.csv = file_path(
        args.csv if args.csv is not None else
        (config["paths"]["csv"] if args.video is not None else None),
        output_dir, config_args.config.resolve(),
    )
    min_window = config["analysis"]["min_window"]
    if (not np.isfinite(args.window) or args.window < min_window
            or not np.isfinite(args.update) or args.update <= 0
            or not (0 < args.min_bpm < args.max_bpm < float("inf"))):
        parser.error(f"window >= analysis.min_window ({min_window:g}), update > 0, "
                     "and 0 < min-bpm < max-bpm are required; values must be finite")
    if args.fps is not None and args.fps <= 0:
        parser.error("fps must be positive")
    if args.motion_limit <= 0:
        parser.error("motion-limit must be positive")
    if args.video and not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    if args.output_video:
        if not args.video:
            parser.error("--output-video requires --video")
        if args.output_video.suffix.lower() != ".mp4":
            parser.error("--output-video must be an .mp4 file")
        if args.output_video.exists():
            parser.error(f"Output video already exists: {args.output_video}")
    estimator = None
    if method_name == "efficientphys":
        model_path = Path(method_cfg.get("model_path", "models/PURE_EfficientPhys.pth")).expanduser()
        if not model_path.is_absolute():
            model_path = config_args.config.resolve().parent / model_path
        if not model_path.is_file():
            parser.error(f"EfficientPhys weights do not exist: {model_path}")
        try:
            estimator = EfficientPhysEstimator(
                model_path=model_path,
                device=str(method_cfg.get("device", "cuda")),
                frame_depth=int(method_cfg.get("frame_depth", 10)),
                image_size=int(method_cfg.get("input_size", 72)),
                min_spectral_score=float(method_cfg.get("min_spectral_score", 0.15)),
                min_window=min_window,
            )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            parser.error(f"Cannot initialize EfficientPhys: {exc}")
    if not hasattr(mp, "solutions"):
        parser.error("This implementation needs mediapipe==0.10.21 (mp.solutions.face_mesh)")
    LOGGER.info("配置: %s", config_args.config.resolve())
    compute = str(estimator.device) if estimator else "CPU"
    LOGGER.info("输入: %s | 方法: %s | 计算: %s", args.video if args.video else f"摄像头 {args.camera}",
                method_name, compute)
    LOGGER.info("分析窗口 %.1fs | 更新间隔 %.1fs | 心率范围 %.0f–%.0f bpm | 日志间隔 %.1fs",
                args.window, args.update, args.min_bpm, args.max_bpm, args.log_interval)
    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        cap.release()
        parser.error("Cannot open video/camera; check source and camera permissions")
    if args.video:
        # Phone recordings can store rotation separately from the encoded pixels.
        # Apply it before tracking and rendering; the exported canvas has no rotation tag.
        orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        auto_rotation = cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        LOGGER.info("视频旋转标记 %.0f° | 自动方向校正 %s", orientation,
                    "已启用" if auto_rotation else "后端不支持")
        if orientation and not auto_rotation:
            cap.release()
            parser.error("Video has rotation metadata but this decoder cannot auto-rotate it")
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
        cap.set(cv2.CAP_PROP_FPS, config["camera"]["fps"])
    nominal = args.fps or cap.get(cv2.CAP_PROP_FPS)
    if args.video and (not np.isfinite(nominal) or nominal <= 0):
        cap.release()
        parser.error("Video has invalid FPS metadata; specify --fps")
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) if args.video else 0
    total_frames = int(frame_count) if np.isfinite(frame_count) and frame_count > 0 else None
    LOGGER.info("输入尺寸 %.0fx%.0f | 名义帧率 %.2f FPS | 总帧数 %s",
                cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                nominal, total_frames if total_frames is not None else "未知")
    if args.video and total_frames and total_frames / nominal < args.window:
        LOGGER.warning("视频名义时长 %.1fs 小于分析窗口 %.1fs，可能无法产生心率结果",
                       total_frames / nominal, args.window)
    samples, neural_samples, history = deque(), deque(), deque(maxlen=config["analysis"]["smoothing_samples"])
    neural_box = None
    trend = HeartRateTrend(seconds=config["analysis"]["trend_seconds"])
    result = None
    displayed = None
    # 仅作候选组接近时的历史参考，与 history 中的显示平滑数据分开管理。
    last_good_bpm, last_good_time = None, None
    previous_points = None
    previous_time = None
    last_estimate = -float("inf")
    last_timestamp = -1.0
    last_valid = None
    window_ready = False
    frame_index = 0
    fallback_warned = False
    csv_file = None
    writer = None
    video_writer = None
    inference_pool = ThreadPoolExecutor(max_workers=1) if estimator and not args.video else None
    pending_estimate = None
    estimates = good_estimates = 0
    written_frames = 0
    last_log = -float("inf")
    stop_reason = "处理异常"
    wall_start = time.perf_counter()
    try:
        if args.output_video:
            args.output_video.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                nominal, SIZE,
            )
            if not video_writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {args.output_video}")
            LOGGER.info("视频输出: %s", args.output_video.resolve())
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_file = args.csv.open("x", newline="", encoding="utf-8-sig")
            writer = csv.writer(csv_file)
            writer.writerow(["time_s", "raw_bpm", "display_bpm", "status", "spectral_score",
                             "sample_fps", "forehead_bpm", "left_bpm", "right_bpm",
                             "fusion_reason", "support_regions", "roi_diagnostics_json"])
            LOGGER.info("CSV 输出: %s", args.csv.resolve())
        LOGGER.info("开始处理，积累满 %.1fs 有效数据后估计心率；Q/Esc 退出，R 重新采集", args.window)
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.6, min_tracking_confidence=0.6,
        ) as mesh:
            cv2.namedWindow("rPPG Demo", cv2.WINDOW_NORMAL)
            while True:
                ok, frame = cap.read()
                if not ok:
                    stop_reason = "视频读取结束" if args.video else "摄像头读取失败"
                    if not args.video or (total_frames and frame_index < total_frames):
                        LOGGER.warning("%s，已读取 %d 帧", stop_reason, frame_index)
                    break
                if args.video:
                    measured = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if args.fps:
                        timestamp = frame_index / nominal
                    elif np.isfinite(measured) and measured > last_timestamp:
                        timestamp = measured
                    else:
                        timestamp = last_timestamp + 1 / nominal if frame_index else 0.0
                        if not fallback_warned:
                            LOGGER.warning("视频时间戳不可用或不递增，退回使用名义帧率 %.2f FPS", nominal)
                            fallback_warned = True
                else:
                    timestamp = time.perf_counter() - wall_start
                last_timestamp = timestamp
                frame_index += 1
                observation = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                status = "NO FACE"
                valid = False
                if observation.multi_face_landmarks:
                    landmarks = observation.multi_face_landmarks[0].landmark
                    points = np.asarray([(p.x * frame.shape[1], p.y * frame.shape[0]) for p in landmarks])
                    neural_crop = None
                    if estimator:
                        if neural_box is None:
                            neural_box = face_box(frame, points, float(method_cfg.get("face_box_scale", 1.5)))
                        neural_crop = face_crop(frame, points, estimator.image_size,
                                                float(method_cfg.get("face_box_scale", 1.5)), neural_box)
                        # EfficientPhys 直接使用放大的人脸裁剪图像；以下占位数据
                        # 用于兼容缓冲和输出流程，不应用 POS 的区域质量门槛。
                        data = (np.zeros((3, 3)), np.zeros(3), np.zeros(3), []) if neural_crop is not None else None
                    else:
                        data = extract_regions(frame, points)
                    status = "FACE / LIGHT INVALID"
                    if data:
                        means, dark, clipped, polygons = data
                        motion = np.zeros(3)
                        if previous_points is not None and previous_time is not None:
                            scale = max(np.ptp(points[:, 0]), 1)
                            motion = np.asarray([
                                np.sqrt(np.mean(np.sum((points[indices]-previous_points[indices])**2, axis=1))) /
                                scale / max(timestamp-previous_time, 1e-6)
                                for indices in REGIONS.values()
                            ])
                        previous_points, previous_time = points, timestamp
                        if last_valid is not None and timestamp - last_valid > 0.25:
                            LOGGER.warning("有效采样间断 %.3fs，重新积累分析窗口", timestamp - last_valid)
                            samples.clear()
                            neural_samples.clear()
                            history.clear()
                            result, displayed = None, None
                            pending_estimate = None
                            window_ready = False
                            neural_box = None
                            last_good_bpm, last_good_time = None, None
                        last_valid = timestamp
                        samples.append((timestamp, means, motion, dark, clipped))
                        if estimator:
                            neural_samples.append((timestamp, neural_crop))
                        while len(samples) > 1 and timestamp - samples[1][0] >= args.window:
                            samples.popleft()
                        while len(neural_samples) > 1 and timestamp - neural_samples[1][0] >= args.window:
                            neural_samples.popleft()
                        valid = True
                        status = "COLLECTING"
                        window_ready = window_ready or timestamp - samples[0][0] >= args.window
                        if window_ready:
                            status = result['status'] if result else "ANALYZING"
                            result_ready = False
                            if pending_estimate is not None and pending_estimate[0].done():
                                result = pending_estimate[0].result()
                                last_estimate = pending_estimate[1]
                                pending_estimate = None
                                result_ready = True
                            elif pending_estimate is None and timestamp - last_estimate >= args.update:
                                prior = last_good_bpm if last_good_time is not None and timestamp-last_good_time <= fusion_cfg["history_seconds"] else None
                                if inference_pool:
                                    pending_estimate = (inference_pool.submit(
                                        estimator.estimate, tuple(neural_samples), args.min_bpm / 60,
                                        args.max_bpm / 60, prior), timestamp)
                                    status = "ANALYZING"
                                else:
                                    last_estimate = timestamp
                                    if estimator:
                                        result = estimator.estimate(neural_samples, args.min_bpm / 60,
                                                                    args.max_bpm / 60, prior)
                                    else:
                                        result = estimate(samples, args.min_bpm / 60, args.max_bpm / 60,
                                                          args.motion_limit, fusion_cfg, prior)
                                    result_ready = True
                            if result_ready:
                                estimates += 1
                                status = result['status'] if result else "TIMING / FPS INVALID"
                                if result and status == "GOOD":
                                    good_estimates += 1
                                    history.append(result['bpm'])
                                    displayed = float(np.median(history))
                                    last_good_bpm, last_good_time = result['bpm'], timestamp
                                else:
                                    history.clear()
                                    displayed = None
                                if writer:
                                    # 此处记录窗口结论；下面的当前帧检查仍可能隐藏显示数字。
                                    # 诊断 JSON 保存实际匹配峰，便于解释它与各区域最高峰的差别。
                                    writer.writerow([round(timestamp, 3), result['bpm'] if result else '',
                                                     displayed if displayed is not None else '', status,
                                                     result['quality'] if result else '', result['fs'] if result else '',
                                                     *(result['rates'] if result else ['', '', '']),
                                                     result['reason'] if result else 'Invalid sampling timeline',
                                                     '|'.join(list(REGIONS)[i] for i in result['support']) if result else '',
                                                     json.dumps([
                                                         dict(region=name, primary_score=float(result['primary_scores'][i]),
                                                              matched_bpm=result['matched_rates'][i],
                                                              matched_score=result['roi_scores'][i],
                                                              motion_fraction=float(result['moving'][i]),
                                                              dark_fraction=float(result['dark'][i]),
                                                              clipped_fraction=float(result['clipped'][i]),
                                                              excluded=result['excluded'][i], peaks=result['candidates'][i])
                                                         for i, name in enumerate(REGIONS)
                                                     ]) if result else ''])
                                    csv_file.flush()
                        # 当前帧质量检查立即影响旧窗口结果的显示。
                        # 旧窗口合格不代表当前画面仍合格；只统计原支持区域中的可用者。
                        if not estimator:
                            current_usable = (motion <= args.motion_limit) & (dark <= 0.4) & (clipped <= 0.3) & np.all(np.isfinite(means), axis=1)
                            if np.count_nonzero(motion > args.motion_limit) >= 2:
                                status = "HOLD STILL"
                            elif np.count_nonzero(dark > 0.4) >= 2:
                                status = "LOW LIGHT"
                            elif np.count_nonzero(clipped > 0.3) >= 2:
                                status = "OVEREXPOSED"
                            elif result and status == "GOOD" and sum(current_usable[i] for i in result['support']) < 2:
                                status = "ROI QUALITY"
                            for polygon, color in zip(polygons, COLORS):
                                cv2.polylines(frame, [cv2.convexHull(polygon)], True, color, 1, cv2.LINE_AA)
                if not valid:
                    previous_points, previous_time = None, None
                    displayed = None
                    history.clear()
                    if last_valid is not None and timestamp - last_valid > 0.25:
                        LOGGER.warning("人脸采样中断 %.3fs，重新积累分析窗口", timestamp - last_valid)
                        samples.clear()
                        neural_samples.clear()
                        result = None
                        pending_estimate = None
                        window_ready = False
                        neural_box = None
                        last_good_bpm, last_good_time = None, None
                progress = samples[-1][0] - samples[0][0] if len(samples) > 1 else 0.0
                visible_bpm = displayed if status == "GOOD" else None
                trend.record(timestamp, visible_bpm, last_estimate)
                canvas = render(frame, result, visible_bpm, status, progress, args, trend, timestamp)
                if video_writer:
                    video_writer.write(canvas)
                    written_frames += 1
                now = time.perf_counter()
                if now - last_log >= args.log_interval:
                    last_log = now
                    bpm_text = f"{displayed:.1f} bpm" if status == "GOOD" and displayed is not None else "--"
                    LOGGER.info(
                        "帧 %d/%s | 源时间 %.1fs | 累计处理速度 %.1f FPS | 缓冲 %.1f/%.1fs | 状态 %s | 心率 %s",
                        frame_index, total_frames if total_frames is not None else "?", timestamp,
                        frame_index / max(now - wall_start, 1e-6), progress, args.window, status, bpm_text,
                    )
                    if result:
                        if estimator:
                            LOGGER.info("EfficientPhys 窗口 t=%.1fs | 状态 %s | 候选 %.1f bpm | 频谱分数 %.2f | 采样率 %.1f Hz",
                                        last_estimate, result['status'], result['bpm'], result['quality'], result['fs'])
                        else:
                            LOGGER.info("最近窗口 t=%.1fs | 状态 %s | 频谱分数 %.2f | 采样率 %.1f Hz | 额头/左颊/右颊候选 %.1f/%.1f/%.1f bpm",
                                        last_estimate, result['status'], result['quality'], result['fs'], *result['rates'])
                            LOGGER.info("融合依据: %s | 支持区域: %s | 各区排除原因: %s",
                                        result['reason'], [list(REGIONS)[i] for i in result['support']], result['excluded'])
                cv2.imshow("rPPG Demo", canvas)
                delay = 1
                if args.video and not video_writer:
                    delay = max(1, min(100, int(1000 * ((frame_index / nominal) - (time.perf_counter() - wall_start)))))
                key = cv2.waitKey(delay) & 0xFF
                if key in (27, ord('q')) or cv2.getWindowProperty("rPPG Demo", cv2.WND_PROP_VISIBLE) < 1:
                    stop_reason = "用户停止"
                    break
                if key == ord('r'):
                    LOGGER.info("用户重置缓冲，重新采集")
                    trend.clear()
                    samples.clear()
                    neural_samples.clear()
                    history.clear()
                    result, displayed, last_valid = None, None, None
                    neural_box = None
                    previous_points, previous_time = None, None
                    last_good_bpm, last_good_time = None, None
                    pending_estimate = None
                    window_ready = False
    except KeyboardInterrupt:
        stop_reason = "用户中断"
        LOGGER.info("收到 Ctrl+C，正在关闭输出文件")
    except Exception:
        LOGGER.exception("处理失败，输出可能不完整")
        raise
    finally:
        if inference_pool:
            inference_pool.shutdown(wait=True)
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()
        elapsed = time.perf_counter() - wall_start
        LOGGER.info("结束: %s | 处理 %d 帧 | 耗时 %.2fs | 平均 %.1f FPS | 有效窗口 %d/%d",
                    stop_reason, frame_index, elapsed, frame_index / max(elapsed, 1e-6),
                    good_estimates, estimates)
        if estimates == 0:
            LOGGER.info("本次未完成心率估计：未积累到完整的有效分析窗口，或在分析前结束")
        if video_writer and written_frames:
            LOGGER.info("视频输出已关闭: %s | 写入 %d 帧（提前停止或异常时仅含已处理部分）",
                        args.output_video.resolve(), written_frames)
        if csv_file:
            LOGGER.info("CSV 输出已关闭: %s", args.csv.resolve())


if __name__ == "__main__":
    main()
