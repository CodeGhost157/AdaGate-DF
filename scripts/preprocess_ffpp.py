#!/usr/bin/env python3
from __future__ import annotations

import argparse

from deepfake_lowres.data.preprocess_ffpp import preprocess_ffpp


def main():
    parser = argparse.ArgumentParser(description="Extract frames and metadata from FaceForensics++ videos.")
    parser.add_argument("--raw-root", required=True, help="Path to the raw FaceForensics++ dataset root.")
    parser.add_argument("--output-root", required=True, help="Output folder. metadata.csv and all_frames/ are written here.")
    parser.add_argument("--target-fps", type=int, default=2)
    parser.add_argument("--max-frames-per-video", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    metadata_csv = preprocess_ffpp(
        raw_root=args.raw_root,
        output_root=args.output_root,
        target_fps=args.target_fps,
        max_frames_per_video=args.max_frames_per_video,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"Metadata saved to: {metadata_csv}")


if __name__ == "__main__":
    main()
