"""Source-tracked ROIs, uint8 pixel modulation, and lossless examples."""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

REGIONS = {
    "forehead": [109, 10, 338, 337, 151, 108],
    "left_cheek": [50, 101, 205, 207, 187, 123],
    "right_cheek": [280, 330, 425, 427, 411, 352],
}
ROI_NAMES = ["pooled", *REGIONS, "background"]


def prepare_geometry(frames, sigma):
    geometry, box = [], None
    height, width = frames.shape[1:3]
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False,
                                       min_detection_confidence=0.6, min_tracking_confidence=0.6) as mesh:
        for index, frame in enumerate(frames):
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            polygons = None
            if result.multi_face_landmarks:
                points = np.asarray([(p.x * width, p.y * height) for p in result.multi_face_landmarks[0].landmark])
                polygons = [np.round(points[ids]).astype(np.int32) for ids in REGIONS.values()]
                if any(np.any(p[:, 0] < 0) or np.any(p[:, 0] >= width) or np.any(p[:, 1] < 0) or np.any(p[:, 1] >= height) for p in polygons):
                    polygons = None
                if box is None and polygons is not None:
                    from efficientphys import face_box
                    box = face_box(frame, points, 1.5)
            geometry.append(polygons)
            if (index + 1) % 300 == 0:
                print("  landmarks {}/{}".format(index + 1, len(frames)), flush=True)
    if box is None:
        raise ValueError("No valid source face; not substituting a whole-frame ROI")
    return geometry, box


def masks(shape, polygons, sigma):
    regions = []
    for polygon in polygons:
        mask = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(polygon), 1)
        regions.append(mask.astype(bool))
    union = np.any(regions, axis=0)
    alpha = union.astype(np.float32)
    if sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return regions, union, alpha


def inject(frame, alpha, value, strength, weights):
    # Weights are specified in RGB, OpenCV pixels are BGR. The formula is a heuristic baseline.
    gain = np.exp(strength * value * alpha[..., None] * np.asarray(weights[::-1], dtype=np.float32))
    return np.rint(np.clip(frame.astype(np.float32) * gain, 0, 255)).astype(np.uint8)


def rgb_means(frame, regions, union, alpha, box):
    means = []
    for mask in [union, *regions]:
        pixels = frame[mask]
        means.append(pixels[:, ::-1].mean(axis=0) if len(pixels) >= 50 else [np.nan] * 3)
    height, width = frame.shape[:2]
    size = max(16, int(min(height, width) * 0.12))
    x0, y0, x1, y1 = box
    # Pick the first corner outside the enlarged source face box and outside the edited support.
    background = None
    for x, y in [(0, 0), (width - size, 0), (0, height - size), (width - size, height - size)]:
        intersects = x < x1 and x + size > x0 and y < y1 and y + size > y0
        if not intersects and not np.any(alpha[y:y + size, x:x + size] > 0):
            background = frame[y:y + size, x:x + size].reshape(-1, 3)[:, ::-1].mean(axis=0)
            break
    means.append(background if background is not None else [np.nan] * 3)
    return means


def lossless_writer(path, fps, shape):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (shape[1], shape[0]))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("FFV1 writer unavailable; will not silently replace it with a lossy codec")
    return writer


def process_variant(frames, geometry, box, fps, cfg, target=None, strength=0.0, weights=None, video_path=None):
    size = cfg["efficientphys"]["image_size"]
    rgb = np.full((len(frames), len(ROI_NAMES), 3), np.nan, dtype=np.float64)
    crops = np.zeros((len(frames), size, size, 3), dtype=np.uint8)
    valid = np.zeros(len(frames), dtype=bool)
    squared, absolute, roi_squared, roi_absolute, pixels, roi_pixels = 0., 0., 0., 0., 0, 0
    outside_squared = 0.
    clipping_events, edit_pixels = 0, 0
    writer = lossless_writer(video_path, fps, frames.shape[1:]) if video_path is not None else None
    crop_left, crop_top, crop_right, crop_bottom = box
    expected_crcs = []
    import zlib
    try:
        for index, source in enumerate(frames):
            polygons = geometry[index]
            edited = source
            if polygons is not None:
                regions, union, alpha = masks(source.shape[:2], polygons, cfg["mask_blur_sigma"])
                if target is not None:
                    edited = inject(source, alpha, target[index], strength, weights)
                rgb[index] = rgb_means(edited, regions, union, alpha, box)
                difference = edited.astype(np.float32) - source
                support = alpha > 0
                outside_squared += float(np.square(difference[~support]).sum())
                squared += float(np.square(difference).sum())
                absolute += float(np.abs(difference).sum())
                roi_squared += float(np.square(difference[union]).sum())
                roi_absolute += float(np.abs(difference[union]).sum())
                pixels += source.size
                roi_pixels += int(union.sum()) * 3
                clipping_events += int(np.sum((edited[support] == 0) | (edited[support] == 255)))
                edit_pixels += int(support.sum()) * 3
                valid[index] = np.isfinite(rgb[index, :4]).all()
            crops[index] = cv2.cvtColor(cv2.resize(edited[crop_top:crop_bottom, crop_left:crop_right], (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            if writer is not None:
                writer.write(edited)
                expected_crcs.append(zlib.crc32(edited.tobytes()))
    finally:
        if writer is not None:
            writer.release()
    verified_frames = 0
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        try:
            while True:
                ok, decoded = cap.read()
                if not ok:
                    break
                if verified_frames >= len(expected_crcs) or zlib.crc32(decoded.tobytes()) != expected_crcs[verified_frames]:
                    raise ValueError("Saved FFV1 example differs from evaluated uint8 frames")
                verified_frames += 1
            if verified_frames != len(frames):
                raise ValueError("Saved FFV1 example is truncated")
        finally:
            cap.release()
    mse = squared / max(pixels, 1)
    visual = {"pixel_mae": absolute / max(pixels, 1), "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
              "roi_pixel_mae": roi_absolute / max(roi_pixels, 1),
              "roi_psnr_db": float(10 * np.log10(255 ** 2 / (roi_squared / max(roi_pixels, 1)))) if roi_squared else None,
              "outside_edit_support_squared_error": outside_squared, "saturated_channel_fraction_in_edit_support": clipping_events / max(edit_pixels, 1),
              "face_frame_coverage": float(valid.mean()), "example_video_verified_frames": verified_frames}
    if outside_squared != 0:
        raise AssertionError("Pixels outside edit support changed")
    return rgb, crops, valid, visual


def save_preview(path, frame, polygons, sigma):
    overlay = frame.copy()
    if polygons is not None:
        for polygon in polygons:
            cv2.polylines(overlay, [polygon], True, (0, 255, 0), 1)
        _, _, alpha = masks(frame.shape[:2], polygons, sigma)
        colored = cv2.applyColorMap((alpha * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        panel = np.concatenate([frame, overlay, colored], axis=1)
    else:
        panel = np.concatenate([frame, overlay, np.zeros_like(frame)], axis=1)
    if not cv2.imwrite(str(path), panel):
        raise RuntimeError("Cannot save ROI preview")
