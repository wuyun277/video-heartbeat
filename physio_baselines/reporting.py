"""CSV outputs and persistent two-document experiment bookkeeping."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone, timedelta
from itertools import groupby
from pathlib import Path

import numpy as np

from .data import ROOT

BEGIN, END = "<!-- AUTO:BEGIN -->", "<!-- AUTO:END -->"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    keys = ["variant", "target", "reference_type", "extractor", "roi"]
    ordered = sorted(rows, key=lambda row: tuple(str(row[k]) for k in keys))
    summaries = []
    for group, values in groupby(ordered, key=lambda row: tuple(row[k] for k in keys)):
        values = list(values)
        valid = [row for row in values if row["status"] == "ok"]
        result = dict(zip(keys, group))
        result.update(total_windows=len(values), valid_windows=len(valid), coverage=len(valid) / len(values), subjects=len({v["subject"] for v in values}))
        for field, name, transform in [("hr_error_bpm", "mae_bpm_subject_mean", abs), ("r_zero_signed", "r_zero_signed_subject_mean", float), ("r_zero_abs", "r_zero_abs_subject_mean", float), ("snr_db", "snr_db_subject_mean", float)]:
            by_subject = []
            for subject in sorted({v["subject"] for v in values}):
                numbers = [transform(v[field]) for v in valid if v["subject"] == subject]
                if numbers:
                    by_subject.append(float(np.mean(numbers)))
            result[name] = float(np.mean(by_subject)) if by_subject else None
        result["subjects_with_valid_windows"] = len({v["subject"] for v in valid})
        summaries.append(result)
    return summaries


def main_table(summaries):
    lines = ["| 方法 | 目标/参考 | 提取器 | 有效窗口 | HR MAE (bpm) | 零延迟绝对相关 |", "|---|---|---|---|---|---|"]
    for row in summaries:
        if row["roi"] != "pooled" and row["extractor"] != "EfficientPhys":
            continue
        def fmt(key):
            value = row.get(key)
            return "—" if value is None else "{:.3f}".format(value)
        lines.append("| {} | {}/{} | {} | {}/{} | {} | {} |".format(row["variant"], row["target"], row["reference_type"], row["extractor"], row["valid_windows"], row["total_windows"], fmt("mae_bpm_subject_mean"), fmt("r_zero_abs_subject_mean")))
    return "\n".join(lines)


def update_documents(run_dir, command, status, details, summaries=None):
    run_dir = Path(run_dir)
    relative = run_dir.relative_to(ROOT).as_posix()
    log = ROOT / "实验记录.md"
    if not log.exists():
        log.write_text("# 实验记录\n\n按运行追加，失败记录保留。时间统一为 Asia/Shanghai（UTC+08:00）。\n", encoding="utf-8")
    entry = "\n## {} · {} · {}\n\n- 命令：`{}`\n- 状态：**{}**\n- 运行目录：`{}`\n- 详情：{}\n".format(now(), run_dir.name, command, command, status, relative, details)
    if summaries:
        entry += "\n" + main_table(summaries) + "\n\n完整窗口、逐 ROI 与带符号波形相关见运行目录 CSV/NPZ；本表按受试者先平均。绝对相关忽略极性，不等于相位或完整生理真实性已验证。\n"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    progress = ROOT / "当前进展与下一步.md"
    if not progress.exists():
        progress.write_text("# 当前进展与下一步\n\n" + BEGIN + "\n" + END + "\n\n## 人工研究判断\n\n本节由实验人员维护；自动命令保留本节。\n", encoding="utf-8")
    existing = progress.read_text(encoding="utf-8")
    latest = "\n更新时间：{}\n\n- 最近操作：`{}`，状态 **{}**。\n- 最近运行：`{}`。\n- 详情：{}\n- 当前阶段：小样本开发验证，尚非论文级完整复现。\n".format(now(), command, status, relative, details)
    if summaries:
        latest += "\n" + main_table(summaries) + "\n"
    if BEGIN not in existing or END not in existing:
        raise ValueError("Progress document is missing automatic section markers")
    prefix, rest = existing.split(BEGIN, 1)
    _, suffix = rest.split(END, 1)
    progress.write_text(prefix + BEGIN + latest + END + suffix, encoding="utf-8")
