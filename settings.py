"""Load the demo's JSON settings and resolve input/output paths."""
from __future__ import annotations

import json
import math
from pathlib import Path


# 默认配置相对于本模块定位，避免切换启动目录后找不到配置文件。
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")


# 读取 UTF-8 JSON 配置，检查必需章节和字段是否齐全。
def load_config(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")
    for section, keys in {
        "paths": ("input_dir", "output_dir", "input_video", "output_video", "csv"),
        "camera": ("index", "width", "height", "fps"),
        "analysis": ("window", "update", "min_bpm", "max_bpm", "motion_limit"),
    }.items():
        if not isinstance(data.get(section), dict) or any(key not in data[section] for key in keys):
            raise ValueError(f"Config section {section!r} is missing required fields")
    # 新增显示参数可省略，兼容此前保存的配置文件。
    analysis = data["analysis"]
    analysis.setdefault("min_window", 8.0)
    min_window = analysis["min_window"]
    if (isinstance(min_window, bool) or not isinstance(min_window, (int, float))
            or not math.isfinite(min_window) or min_window <= 0):
        raise ValueError("analysis.min_window must be finite and positive")
    analysis.setdefault("smoothing_samples", 3)
    analysis.setdefault("trend_seconds", 60.0)
    smoothing = analysis["smoothing_samples"]
    if isinstance(smoothing, bool) or not isinstance(smoothing, int) or smoothing < 1:
        raise ValueError("analysis.smoothing_samples must be a positive integer")
    trend_seconds = analysis["trend_seconds"]
    if (isinstance(trend_seconds, bool) or not isinstance(trend_seconds, (int, float))
            or not math.isfinite(trend_seconds) or trend_seconds <= 0):
        raise ValueError("analysis.trend_seconds must be finite and positive")
    return data


# 展开用户目录，并以配置文件所在目录解析相对目录。
def directory(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_path.parent / path


# 区分绝对路径、单独文件名和带子目录的相对路径。
def file_path(value: str | Path | None, folder: Path, config_path: Path) -> Path | None:
    # 单独文件名使用输入/输出目录；带子目录的相对路径使用配置文件所在目录。
    # 因此 output/result.mp4 不会被重复解析成 output/output/result.mp4。
    if value is None or value == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return folder / path
    return config_path.parent / path
