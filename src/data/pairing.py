"""Pair CASIA2 authentic/tampered images with ground-truth masks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MASK_EXTS = {".png"}
SKIP_NAMES = {"_list.txt", "thumbs.db"}


@dataclass(frozen=True)
class Sample:
    stem: str
    label: int
    orig_image: Path
    orig_mask: Path | None
    tamper_kind: str
    match_method: str | None


def _is_skipped(path: Path) -> bool:
    return path.name.lower() in SKIP_NAMES


def iter_images(folder: Path, suffixes: set[str] = IMAGE_EXTS) -> list[Path]:
    files: list[Path] = []
    for path in folder.iterdir():
        if not path.is_file() or _is_skipped(path):
            continue
        if path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files)


def gt_image_stem(gt_stem: str) -> tuple[str, str]:
    """Map a mask filename stem back to the image stem and match tag."""
    if gt_stem.endswith("_gt3"):
        return gt_stem[: -len("_gt3")], "gt3"
    if gt_stem.endswith("_gt"):
        return gt_stem[: -len("_gt")], "exact"
    return gt_stem, "exact"


def trailing_id(stem: str) -> str:
    return stem.rsplit("_", 1)[-1]


def tamper_kind_from_stem(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "Tp" and parts[1] in {"D", "S"}:
        return parts[1]
    return ""


def _index_masks(gt_dir: Path) -> dict[str, tuple[Path, str]]:
    indexed: dict[str, tuple[Path, str]] = {}
    for path in iter_images(gt_dir, MASK_EXTS):
        image_stem, method = gt_image_stem(path.stem)
        if image_stem in indexed:
            raise ValueError(f"Duplicate ground-truth stem {image_stem}: {path}")
        indexed[image_stem] = (path, method)
    return indexed


def pair_casia2(src: Path) -> tuple[list[Sample], dict[str, Any]]:
    """Pair Au/Tp images with masks. Authentic samples have orig_mask=None."""
    src = src.resolve()
    au_dir = src / "Au"
    tp_dir = src / "Tp"
    gt_dir = src / "CASIA 2 Groundtruth"

    authentic = iter_images(au_dir)
    tampered = iter_images(tp_dir)
    mask_index = _index_masks(gt_dir)

    samples: list[Sample] = []
    for path in authentic:
        samples.append(
            Sample(
                stem=path.stem,
                label=0,
                orig_image=path,
                orig_mask=None,
                tamper_kind="",
                match_method=None,
            )
        )

    used_mask_stems: set[str] = set()
    exact_matches: list[dict[str, str]] = []
    gt3_matches: list[dict[str, str]] = []
    unmatched_tp: list[Path] = []

    for path in tampered:
        hit = mask_index.get(path.stem)
        if hit is None:
            unmatched_tp.append(path)
            continue
        mask_path, method = hit
        used_mask_stems.add(path.stem)
        sample = Sample(
            stem=path.stem,
            label=1,
            orig_image=path,
            orig_mask=mask_path,
            tamper_kind=tamper_kind_from_stem(path.stem),
            match_method=method,
        )
        samples.append(sample)
        record = {"image": path.name, "mask": mask_path.name, "method": method}
        if method == "gt3":
            gt3_matches.append(record)
        else:
            exact_matches.append(record)

    leftover_masks = {
        stem: (path, method)
        for stem, (path, method) in mask_index.items()
        if stem not in used_mask_stems
    }
    leftover_by_id: dict[str, list[tuple[str, Path]]] = {}
    for stem, (path, _method) in leftover_masks.items():
        leftover_by_id.setdefault(trailing_id(stem), []).append((stem, path))

    id_fallback: list[dict[str, str]] = []
    unpaired_images: list[str] = []
    ambiguous: list[dict[str, Any]] = []

    still_unmatched: list[Path] = []
    for path in unmatched_tp:
        image_id = trailing_id(path.stem)
        candidates = leftover_by_id.get(image_id, [])
        if len(candidates) == 1:
            mask_stem, mask_path = candidates[0]
            leftover_by_id.pop(image_id, None)
            leftover_masks.pop(mask_stem, None)
            samples.append(
                Sample(
                    stem=path.stem,
                    label=1,
                    orig_image=path,
                    orig_mask=mask_path,
                    tamper_kind=tamper_kind_from_stem(path.stem),
                    match_method="id_fallback",
                )
            )
            id_fallback.append(
                {
                    "image": path.name,
                    "mask": mask_path.name,
                    "id": image_id,
                    "method": "id_fallback",
                }
            )
        elif len(candidates) > 1:
            ambiguous.append(
                {
                    "image": path.name,
                    "id": image_id,
                    "masks": [mask.name for _stem, mask in candidates],
                }
            )
            still_unmatched.append(path)
        else:
            still_unmatched.append(path)

    unpaired_images = [path.name for path in still_unmatched]
    unpaired_masks = [path.name for path, _method in leftover_masks.values()]

    audit = {
        "n_authentic": len(authentic),
        "n_tampered": len(tampered),
        "n_masks": len(mask_index),
        "n_paired_tampered": sum(1 for sample in samples if sample.label == 1),
        "exact": exact_matches,
        "gt3": gt3_matches,
        "id_fallback": id_fallback,
        "unpaired_images": unpaired_images,
        "unpaired_masks": unpaired_masks,
        "ambiguous": ambiguous,
    }
    return samples, audit


def write_pairing_audit(audit: dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        **audit,
        "n_exact": len(audit["exact"]),
        "n_gt3": len(audit["gt3"]),
        "n_id_fallback": len(audit["id_fallback"]),
    }
    dest.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
