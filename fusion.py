"""Experimental multi-region spectral fusion; thresholds require calibration."""
import numpy as np
from scipy import signal

# 实验性默认值，不是经过校准的概率或 POS 论文规定的通用门槛。
DEFAULTS = {
    "agreement_bpm": 8.0,
    "peak_bandwidth_bpm": 6.0,
    "min_peak_relative_height": 0.25,
    "min_region_score": 0.20,
    "min_fused_score": 0.40,
    "ambiguity_ratio": 0.90,
    "history_seconds": 5.0,
    "max_motion_fraction": 0.15,
}


# 合并默认融合参数并校验范围，拒绝未知配置项。
def options(overrides=None):
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("fusion must be a JSON object")
    values = {**DEFAULTS, **(overrides or {})}
    if set(values) != set(DEFAULTS):
        raise ValueError("Unknown fusion setting")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"fusion.{key} must be a finite positive number")
        if key not in ("agreement_bpm", "peak_bandwidth_bpm", "history_seconds") and value > 1:
            raise ValueError(f"fusion.{key} must be at most 1")
    return values


# 匹配各区域的实测频谱峰，每个区域对同一候选最多贡献一票。
def fuse_spectra(frequencies, spectra, eligible, low, high, cfg, previous_bpm=None):
    """Match measured primary/secondary peaks, never invent half/double frequencies.

    Each region contributes at most one vote per candidate. History only breaks
    near ties with equal support; it cannot create support or validate a single ROI.
    """
    # 各区域先归一化频段内功率，避免幅度较大的区域仅凭振幅占据优势。
    band = (frequencies >= low) & (frequencies <= high)
    bpm_axis = frequencies[band] * 60
    candidates = []
    for power, usable in zip(spectra, eligible):
        p = np.asarray(power)[band]
        peaks = []
        if usable and np.all(np.isfinite(p)) and p.sum() > 1e-20:
            normalized = p / p.sum()
            # 两端补零使搜索边界的峰也能成为候选，随后限制高度、积分功率和峰间距。
            maxima = signal.find_peaks(np.r_[0.0, normalized, 0.0])[0] - 1
            for index in sorted(maxima, key=lambda i: normalized[i], reverse=True):
                rate = float(bpm_axis[index])
                mass = float(normalized[np.abs(bpm_axis-rate) <= cfg["peak_bandwidth_bpm"]].sum())
                if normalized[index] < normalized.max()*cfg["min_peak_relative_height"] or mass < cfg["min_region_score"]:
                    continue
                if any(abs(rate-item[0]) < cfg["peak_bandwidth_bpm"] for item in peaks):
                    continue
                peaks.append((rate, mass))
                if len(peaks) == 3:
                    break
        candidates.append(peaks)

    seeds = [peak[0] for region in candidates for peak in region]
    # 将间距在一致性容差内的跨区域峰值中点也作为聚类种子。
    for a in range(len(candidates)):
        for b in range(a+1, len(candidates)):
            seeds.extend((pa[0]+pb[0])/2 for pa in candidates[a] for pb in candidates[b]
                         if abs(pa[0]-pb[0]) <= cfg["agreement_bpm"])
    # 围绕种子取半径 agreement_bpm/2 的峰，确保组内最大跨度不超过配置值。
    clusters = []
    for seed in seeds:
        selected = []
        for region, peaks in enumerate(candidates):
            nearby = [peak for peak in peaks if abs(peak[0]-seed) <= cfg["agreement_bpm"]/2]
            if nearby:
                rate, mass = max(nearby, key=lambda p: p[1])
                selected.append((region, rate, mass))
        if not selected:
            continue
        weights = np.asarray([p[2] for p in selected])
        fused = float(np.average([p[1] for p in selected], weights=weights))
        clusters.append(dict(bpm=fused, quality=float(weights.mean()), selected=selected))
    empty = dict(bpm=None, quality=0.0, support=[], matched_rates=[None]*3,
                 roi_scores=[0.0]*3, candidates=candidates)
    if not clusters:
        return dict(empty, status="WEAK SIGNAL", reason="No usable spectral peaks")
    # 一个区域一票，优先支持区域数，其次平均质量；后续仍需通过最低质量门槛。
    clusters.sort(key=lambda c: (len(c["selected"]), c["quality"]), reverse=True)
    best = clusters[0]
    rivals = [c for c in clusters if len(c["selected"]) == len(best["selected"])
              and c["quality"] >= best["quality"]*cfg["ambiguity_ratio"]]
    ambiguous = any(abs(c["bpm"]-best["bpm"]) > cfg["agreement_bpm"] for c in rivals)
    history_used = False
    # 历史只在当前实测证据接近时辅助消歧，不产生新峰，也不强制沿用旧心率。
    if ambiguous and previous_bpm is not None:
        close = [c for c in rivals if abs(c["bpm"]-previous_bpm) <= cfg["agreement_bpm"]]
        if close and max(c["bpm"] for c in close)-min(c["bpm"] for c in close) <= cfg["agreement_bpm"]:
            best = min(close, key=lambda c: abs(c["bpm"]-previous_bpm))
            ambiguous, history_used = False, True
    support = [p[0] for p in best["selected"]]
    matched, scores = [None]*3, [0.0]*3
    for region, rate, mass in best["selected"]:
        matched[region], scores[region] = rate, mass
    if len(support) < 2:
        status, reason = "SINGLE ROI", "Only one region supports this frequency"
    elif ambiguous:
        status, reason = "AMBIGUOUS", "Multiple supported frequencies; possible harmonic or noise"
    elif best["quality"] < cfg["min_fused_score"]:
        status, reason = "WEAK SIGNAL", "Shared frequency has insufficient spectral concentration"
    else:
        status, reason = "GOOD", "Shared measured peaks" + ("; history broke a near tie" if history_used else "")
    return dict(bpm=best["bpm"], quality=best["quality"], support=support,
                matched_rates=matched, roi_scores=scores, candidates=candidates,
                status=status, reason=reason)
