from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import shutil
import sys
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

from .data import ROOT, digest, interpolate_labels, inventory, load_clip, load_config
from .reporting import main_table, now, summarize, update_documents, write_csv, write_json
from .signals import NeuralReader, bandpass, extract, metrics, standardize
from .video import ROI_NAMES, prepare_geometry, process_variant, save_preview


def new_run(cfg, command):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    folder = Path(cfg["runs_dir"]) / (stamp + "_" + command + "_" + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", cfg)
    source = folder / "source"
    source.mkdir()
    hashes = {}
    for path in sorted(ROOT.glob("*.py")):
        shutil.copy2(path, source / path.name)
        hashes[path.name] = digest(path)
    model_module = ROOT.parent / "efficientphys.py"
    shutil.copy2(model_module, source / "existing_efficientphys.py")
    hashes["existing_efficientphys.py"] = digest(model_module)
    versions = {}
    for name in ("numpy", "scipy", "opencv-contrib-python", "mediapipe", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    write_json(folder / "environment.json", {"created_at": now(), "python": sys.version, "executable": sys.executable,
               "platform": platform.platform(), "versions": versions, "source_sha256": hashes,
               "model_weights_sha256": digest(cfg["efficientphys"]["weights"]) if "EfficientPhys" in cfg["extractors"] else None})
    write_json(folder / "status.json", {"status": "running", "command": command, "started_at": now()})
    return folder


def target_signals(cfg, times, fps):
    targets = {}
    for target in cfg["targets"]:
        if target["kind"] == "sine":
            values = np.sin(2 * np.pi * target["bpm"] / 60 * (times - times[0]))
        else:
            values, _ = interpolate_labels(Path(cfg["data_dir"]) / target["subject"] / "ground_truth.txt", times)
            values = bandpass(values, fps, cfg["band_hz"])
        targets[target["name"]] = standardize(values)
    return targets


def evaluate(cfg, subject, variant, target_name, rgb, crops, valid, times, fps, references, device_hr, neural):
    rows, predictions = [], {}
    window = int(round(cfg["window_seconds"] * fps))
    neural_error = None
    if neural is not None:
        try:
            predictions["EfficientPhys_full_face"] = neural.predict(crops)
        except Exception as exc:
            neural_error = str(exc)
    for method in cfg["extractors"]:
        rois = ["full_face"] if method == "EfficientPhys" else ROI_NAMES
        for roi in rois:
            key = method + "_" + roi
            stored = predictions.setdefault(key, np.full(len(times), np.nan))
            for start in range(0, len(times) - window + 1, window):
                end = start + window
                base = {"subject": subject, "variant": variant, "extractor": method, "roi": roi,
                        "start_seconds": float(times[start]), "end_seconds_exclusive": float(times[start] + window / fps),
                        "fps": fps, "window_frames": window, "face_frame_coverage": float(valid[start:end].mean())}
                error = None
                try:
                    if method == "EfficientPhys":
                        if neural_error:
                            raise ValueError(neural_error)
                        if not valid[start:end].all():
                            raise ValueError("face detection missing in window")
                        predicted = stored[start:end]
                    else:
                        predicted = extract(rgb[start:end, ROI_NAMES.index(roi)], fps, method, cfg["band_hz"])
                        stored[start:end] = predicted
                except Exception as exc:
                    error = str(exc)
                for name, kind, truth in references:
                    row = dict(base, target=name, reference_type=kind)
                    try:
                        if error:
                            raise ValueError(error)
                        row.update(metrics(predicted, truth[start:end], fps, cfg["band_hz"], cfg["edge_trim_seconds"]))
                        if kind == "source_ppg":
                            row["mean_device_hr"] = float(device_hr[start:end].mean())
                            row["device_hr_error_bpm"] = row["predicted_bpm"] - row["mean_device_hr"]
                        row.update(status="ok", reason="")
                    except Exception as exc:
                        row.update(status="invalid", reason=str(exc))
                    rows.append(row)
    return rows, predictions


def run_experiment(cfg, folder, original_only=False):
    cv2.setNumThreads(2)
    records = inventory(cfg)
    write_json(folder / "inventory.json", records)
    write_csv(folder / "inventory.csv", records)
    bad = [r for r in records if r["subject"] in cfg["subjects"] and r["status"] != "ok"]
    if bad:
        raise ValueError("Selected data failed inventory: " + str(bad))
    neural = NeuralReader(cfg["efficientphys"]) if "EfficientPhys" in cfg["extractors"] else None
    if neural is not None:
        write_json(folder / "neural_protocol.json", {"device": neural.device, "weights": cfg["efficientphys"]["weights"],
            "preprocessing": "fixed first source face box x1.5; RGB resize; whole-clip scalar mean/std; independent chunks with duplicated final frame",
            "postprocessing": "concatenate difference predictions, cumulative sum, common bandpass in each evaluation window",
            "claim": "local evaluation using existing architecture and PURE weights; not an exact paper benchmark reproduction"})
    all_rows, visual_rows, coverage = [], [], []
    for subject in cfg["subjects"]:
        print("Preparing " + subject, flush=True)
        frames, times, fps, source_ppg, source_hr = load_clip(cfg, subject)
        subject_dir = folder / subject
        subject_dir.mkdir()
        geometry, box = prepare_geometry(frames, cfg["mask_blur_sigma"])
        first_valid = next(i for i, p in enumerate(geometry) if p is not None)
        save_preview(subject_dir / "roi_preview.png", frames[first_valid], geometry[first_valid], cfg["mask_blur_sigma"])
        targets = target_signals(cfg, times, fps)
        polygons = np.full((len(frames), 3, 6, 2), -1, dtype=np.int32)
        for i, item in enumerate(geometry):
            if item is not None:
                polygons[i] = item
        np.savez_compressed(subject_dir / "source_geometry.npz", polygons=polygons, face_box=np.array(box), times=times)
        reference_original = [("source", "source_ppg", source_ppg)] + [(name, "target_ppg", value) for name, value in targets.items()]
        variants = [("B0_original", "none", None, 0., None)]
        if not original_only:
            for name, target in targets.items():
                for strength in cfg["strengths"]:
                    for label, weights in [("B1_green", [0., 1., 0.]), ("B2_rgb", cfg["rgb_weights"])]:
                        variants.append(("{}_a{:g}".format(label, strength), name, target, strength, weights))
        for variant, target_name, target, strength, weights in variants:
            print("  {} {}".format(variant, target_name), flush=True)
            video_path = None
            if cfg["save_example_videos"] and subject == cfg["subjects"][0]:
                if target is None or (target_name == next(iter(targets)) and strength == max(cfg["strengths"])):
                    video_path = subject_dir / (variant + "_" + target_name + ".avi")
            rgb, crops, valid, visual = process_variant(frames, geometry, box, fps, cfg, target, strength, weights, video_path)
            references = reference_original if target is None else [(target_name, "target_ppg", target)]
            rows, predictions = evaluate(cfg, subject, variant, target_name, rgb, crops, valid, times, fps, references, source_hr, neural)
            all_rows.extend(rows)
            visual_rows.append(dict(subject=subject, variant=variant, target=target_name, strength=strength, **visual))
            safe_predictions = {key: value for key, value in predictions.items()}
            np.savez_compressed(subject_dir / (variant + "_" + target_name + "_signals.npz"),
                                times=times, fps=fps, rgb=rgb, face_valid=valid, source_ppg=source_ppg, source_device_hr=source_hr,
                                **{"target_" + k: v for k, v in targets.items()}, **safe_predictions)
            del crops
            write_csv(folder / "window_metrics.csv", all_rows)
            write_csv(folder / "visual_metrics.csv", visual_rows)
            write_json(folder / "status.json", {"status": "running", "current_subject": subject,
                       "completed_variant": variant, "target": target_name, "windows_saved": len(all_rows), "updated_at": now()})
        window = round(cfg["window_seconds"] * fps)
        coverage.append({"subject": subject, "frames": len(frames), "fps": fps,
                         "window_frames": window, "full_windows": len(frames) // window,
                         "tail_frames_not_evaluated": len(frames) % window,
                         "source_face_frames": sum(p is not None for p in geometry), "source_face_box": list(box)})
        write_json(folder / "clip_metadata.json", coverage)
        del frames
    summary = summarize(all_rows)
    write_csv(folder / "summary.csv", summary)
    write_json(folder / "summary.json", summary)
    (folder / "summary.md").write_text("# 开发实验汇总\n\n" + main_table(summary) + "\n\n这是小样本像素域基线；无损示例的所有帧均经过解码校验。未实施有损压缩压力测试。\n", encoding="utf-8")
    return summary, visual_rows


def main():
    parser = argparse.ArgumentParser(description="方向一：开发数据检查、像素域 baseline 与实验记录")
    parser.add_argument("command", choices=["inspect", "run", "report"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--clip-seconds", type=float)
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--no-example-videos", action="store_true")
    parser.add_argument("--original-only", action="store_true")
    parser.add_argument("--run-dir", type=Path, help="report: regenerate a completed run's summary and update both documents")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.subjects:
        if set(args.subjects) & set(cfg["reserved_subjects"]):
            parser.error("Reserved subjects cannot be selected for development")
        if len(set(args.subjects)) != len(args.subjects) or any(Path(s).name != s or s in (".", "..") for s in args.subjects):
            parser.error("Subjects must be unique folder names")
        cfg["subjects"] = args.subjects
    if args.clip_seconds is not None:
        if not np.isfinite(args.clip_seconds) or args.clip_seconds < cfg["window_seconds"]:
            parser.error("Clip duration must be finite and at least one window")
        cfg["clip_seconds"] = args.clip_seconds
    if args.no_neural:
        cfg["extractors"] = [x for x in cfg["extractors"] if x != "EfficientPhys"]
    if args.no_example_videos:
        cfg["save_example_videos"] = False
    if args.command == "report":
        if args.run_dir is None:
            parser.error("report requires --run-dir")
        folder = args.run_dir.resolve()
        folder.relative_to(Path(cfg["runs_dir"]).resolve())
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "completed":
            parser.error("Report requires a completed run; partial failures stay marked as failures")
        with (folder / "window_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key in ("hr_error_bpm", "r_zero_signed", "r_zero_abs", "snr_db"):
                if row.get(key):
                    row[key] = float(row[key])
        summary = summarize(rows)
        write_csv(folder / "summary.csv", summary)
        write_json(folder / "summary.json", summary)
        update_documents(folder, "report", "completed", "从已有逐窗口结果重新汇总，未重新运行视频实验。", summary)
        print(folder)
        return
    folder = new_run(cfg, args.command)
    print("Run directory: " + str(folder), flush=True)
    try:
        if args.command == "inspect":
            records = inventory(cfg)
            write_csv(folder / "inventory.csv", records)
            write_json(folder / "inventory.json", records)
            failed = [r for r in records if r["status"] != "ok"]
            if failed:
                raise ValueError("Inventory errors: " + str(failed))
            details = "检查 {} 位受试者的视频元数据、首帧与完整标签；开发 {}；保留 {}。清单检查不等于全视频逐帧验证。".format(len(records), ", ".join(cfg["subjects"]), ", ".join(cfg["reserved_subjects"]))
            summary = None
        else:
            cfg["original_only"] = args.original_only
            write_json(folder / "config.json", cfg)
            summary, visual_rows = run_experiment(cfg, folder, args.original_only)
            details = "开发受试者 {}；每段 {} 秒；提取器 {}；{}；窗口 {} 秒；逐窗口、原始时序和视觉误差均已保存。没有训练新模型。".format(", ".join(cfg["subjects"]), cfg["clip_seconds"], ", ".join(cfg["extractors"]), "只评估原视频" if args.original_only else "B0/B1/B2，目标 " + ", ".join(t["name"] for t in cfg["targets"]), cfg["window_seconds"])
        write_json(folder / "status.json", {"status": "completed", "command": args.command, "finished_at": now(), "details": details})
        update_documents(folder, args.command, "completed", details, summary)
        print("Completed: " + str(folder), flush=True)
    except Exception as exc:
        (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(folder / "status.json", {"status": "failed", "command": args.command, "finished_at": now(), "error": str(exc)})
        update_documents(folder, args.command, "failed", str(exc))
        raise
