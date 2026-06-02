from __future__ import annotations

import csv
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm.auto import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def list_all_videos(root: str | Path):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def infer_label_from_path(path: str | Path):
    s = str(path).lower()
    real_keywords = ["original", "original_sequences", "youtube", "pristine", "real"]
    fake_keywords = ["manipulated", "deepfakes", "face2face", "faceswap", "neuraltextures", "fake", "faceshifter", "deepfakedetection"]
    if any(k in s for k in fake_keywords):
        return 1, "Fake"
    if any(k in s for k in real_keywords):
        return 0, "Real"
    return None, "Unknown"


def build_video_list(root: str | Path):
    rows = []
    for video_path in list_all_videos(root):
        label, label_name = infer_label_from_path(video_path)
        if label is None:
            continue
        rows.append({"video_path": str(video_path), "label": label, "label_name": label_name})
    return rows


def stratified_split(rows, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    rng = random.Random(seed)
    real = [r for r in rows if r["label"] == 0]
    fake = [r for r in rows if r["label"] == 1]
    rng.shuffle(real)
    rng.shuffle(fake)

    def split_one(items):
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    r_train, r_val, r_test = split_one(real)
    f_train, f_val, f_test = split_one(fake)
    out = []
    for split, items in [("Train", r_train + f_train), ("Val", r_val + f_val), ("Test", r_test + f_test)]:
        rng.shuffle(items)
        for row in items:
            out.append({**row, "split": split})
    return out


def safe_rel_video_id(video_path: str | Path, raw_root: str | Path):
    video_path = Path(video_path)
    raw_root = Path(raw_root)
    try:
        rel = video_path.relative_to(raw_root)
    except ValueError:
        rel = video_path.name
    return str(rel).replace("/", "__").replace("\\", "__").replace(".", "_")


def extract_frames_from_video(row, raw_root, frames_dir, target_fps=2, max_frames_per_video=8, jpeg_quality=95):
    video_path = Path(row["video_path"])
    frames_dir = Path(frames_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_step = max(1, int(round(fps / target_fps)))
    video_id = safe_rel_video_id(video_path, raw_root)
    saved = []
    frame_idx = 0
    sample_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            frame_name = f"{row['split']}_{row['label']}_{video_id}_{sample_idx:04d}.jpg"
            frame_path = frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            saved.append({
                "frame_path": str(frame_path),
                "frame_name": frame_name,
                "video_path": str(video_path),
                "video_id": video_id,
                "split": row["split"],
                "label": int(row["label"]),
                "label_name": row["label_name"],
                "frame_index_in_video": int(frame_idx),
                "sample_index": int(sample_idx),
            })
            sample_idx += 1
            if sample_idx >= max_frames_per_video:
                break
        frame_idx += 1
    cap.release()
    return saved


def save_metadata(rows, metadata_csv: str | Path):
    metadata_csv = Path(metadata_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_path", "frame_name", "video_path", "video_id", "split", "label", "label_name", "frame_index_in_video", "sample_index"]
    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_ffpp(raw_root: str | Path, output_root: str | Path, target_fps=2, max_frames_per_video=8, jpeg_quality=95, seed=42, num_workers=4):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    frames_dir = output_root / "all_frames"
    metadata_csv = output_root / "metadata.csv"
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos = stratified_split(build_video_list(raw_root), seed=seed)
    all_rows = []
    if num_workers <= 1:
        for row in tqdm(videos, desc="Extracting frames"):
            all_rows.extend(extract_frames_from_video(row, raw_root, frames_dir, target_fps, max_frames_per_video, jpeg_quality))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(extract_frames_from_video, row, raw_root, frames_dir, target_fps, max_frames_per_video, jpeg_quality) for row in videos]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting frames"):
                all_rows.extend(fut.result())
    save_metadata(all_rows, metadata_csv)
    return metadata_csv
