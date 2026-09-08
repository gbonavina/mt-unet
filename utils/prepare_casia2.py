#!/usr/bin/env python3
"""Pair, split, and export CASIA2 as 256x256 RGB/mask PNGs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pairing import Sample, pair_casia2, write_pairing_audit


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "CASIA2")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "casia2")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def _rel(path: Path | None, src: Path) -> str:
    if path is None:
        return ""
    return path.resolve().relative_to(src.resolve()).as_posix()


def stratified_splits(
    samples: list[Sample],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[Sample]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {total}")

    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    indices = np.arange(len(samples))
    rest_ratio = val_ratio + test_ratio
    train_idx, rest_idx = train_test_split(
        indices,
        test_size=rest_ratio,
        stratify=labels,
        random_state=seed,
    )
    rest_labels = labels[rest_idx]
    test_share = test_ratio / rest_ratio
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_share,
        stratify=rest_labels,
        random_state=seed,
    )
    grouped = {"train": train_idx, "val": val_idx, "test": test_idx}
    return {
        split: [samples[i] for i in idx.tolist()]
        for split, idx in grouped.items()
    }


def export_one(job: tuple[str, str | None, str, str, int, int]) -> str:
    orig_image, orig_mask, out_image, out_mask, size, label = job
    image = Image.open(orig_image).convert("RGB").resize((size, size), Image.BILINEAR)
    Path(out_image).parent.mkdir(parents=True, exist_ok=True)
    image.save(out_image, format="PNG")

    if label == 0 or not orig_mask:
        mask = Image.new("L", (size, size), 0)
    else:
        raw = Image.open(orig_mask).convert("L")
        raw = raw.resize((size, size), Image.NEAREST)
        array = np.asarray(raw)
        binary = np.where(array > 127, 255, 0).astype(np.uint8)
        mask = Image.fromarray(binary, mode="L")
    Path(out_mask).parent.mkdir(parents=True, exist_ok=True)
    mask.save(out_mask, format="PNG")
    return Path(out_image).stem


def rows_for_split(
    split: str,
    split_samples: list[Sample],
    src: Path,
    out: Path,
    size: int,
) -> tuple[list[dict[str, object]], list[tuple[str, str | None, str, str, int, int]]]:
    rows: list[dict[str, object]] = []
    jobs: list[tuple[str, str | None, str, str, int, int]] = []
    for sample in split_samples:
        image_rel = Path("images") / split / f"{sample.stem}.png"
        mask_rel = Path("masks") / split / f"{sample.stem}.png"
        rows.append(
            {
                "stem": sample.stem,
                "split": split,
                "label": sample.label,
                "image_path": image_rel.as_posix(),
                "mask_path": mask_rel.as_posix(),
                "orig_image": _rel(sample.orig_image, src),
                "orig_mask": _rel(sample.orig_mask, src),
                "tamper_kind": sample.tamper_kind,
            }
        )
        orig_mask = None if sample.orig_mask is None else str(sample.orig_mask)
        jobs.append(
            (
                str(sample.orig_image),
                orig_mask,
                str(out / image_rel),
                str(out / mask_rel),
                size,
                sample.label,
            )
        )
    return rows, jobs


def split_counts(split_samples: dict[str, list[Sample]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split, items in split_samples.items():
        labels = Counter(sample.label for sample in items)
        kinds = Counter(sample.tamper_kind or "authentic" for sample in items)
        counts[split] = {
            "n": len(items),
            "authentic": labels.get(0, 0),
            "tampered": labels.get(1, 0),
            "tamper_D": kinds.get("D", 0),
            "tamper_S": kinds.get("S", 0),
        }
    return counts


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    out = args.out.resolve()
    if not src.is_dir():
        raise SystemExit(f"CASIA2 root not found: {src}")

    samples, audit = pair_casia2(src)
    tampered = [sample for sample in samples if sample.label == 1]
    if audit["unpaired_images"] or audit["unpaired_masks"] or audit["ambiguous"]:
        print("Warning: unpaired leftover files remain; see pairing_audit.json", file=sys.stderr)

    out.mkdir(parents=True, exist_ok=True)
    write_pairing_audit(audit, out / "pairing_audit.json")

    grouped = stratified_splits(
        samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    all_rows: list[dict[str, object]] = []
    all_jobs: list[tuple[str, str | None, str, str, int, int]] = []
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "masks" / split).mkdir(parents=True, exist_ok=True)
        rows, jobs = rows_for_split(split, grouped[split], src, out, args.size)
        all_rows.extend(rows)
        all_jobs.extend(jobs)

    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in tqdm(
            pool.map(export_one, all_jobs, chunksize=16),
            total=len(all_jobs),
            desc="export",
        ):
            pass

    frame = pd.DataFrame(all_rows)
    column_order = [
        "stem",
        "split",
        "label",
        "image_path",
        "mask_path",
        "orig_image",
        "orig_mask",
        "tamper_kind",
    ]
    frame = frame[column_order]
    for split in SPLITS:
        split_frame = frame[frame["split"] == split].sort_values("stem")
        split_frame.to_csv(out / f"{split}.csv", index=False)

    counts = split_counts(grouped)
    summary = {
        "src": str(src),
        "out": str(out),
        "size": args.size,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "n_authentic": audit["n_authentic"],
        "n_tampered": audit["n_tampered"],
        "n_paired_tampered": len(tampered),
        "n_exact": len(audit["exact"]),
        "n_gt3": len(audit["gt3"]),
        "n_id_fallback": len(audit["id_fallback"]),
        "unpaired_images": audit["unpaired_images"],
        "unpaired_masks": audit["unpaired_masks"],
        "ambiguous": audit["ambiguous"],
        "splits": counts,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
