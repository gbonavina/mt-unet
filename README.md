# mt-unet

CASIA2 preprocessing for a multitask U-Net (authentic vs tampered classification + tamper-region segmentation). This repo currently prepares the data; the model is left for you to implement.

## Setup

```bash
uv sync
```

Raw CASIA2 lives in `data/CASIA2/` (`Au/`, `Tp/`, `CASIA 2 Groundtruth/`). `data/` is gitignored.

## Preprocess

```bash
uv run python utils/prepare_casia2.py --src data/CASIA2 --out data/processed/casia2 --size 256 --seed 42
```

This pairs every tampered image to its mask (exact filename, `_gt3`, then trailing-id fallback for CASIA2 naming bugs), writes all-zero masks for authentic images, resizes to 256×256 RGB / binary PNG, and splits **70 / 15 / 15** stratified on authentic vs tampered.

## Output

```
data/processed/casia2/
  images/{train,val,test}/<stem>.png
  masks/{train,val,test}/<stem>.png
  train.csv  val.csv  test.csv
  summary.json
  pairing_audit.json
```

CSV columns: `stem, split, label, image_path, mask_path, orig_image, orig_mask, tamper_kind`.

- `label`: `0` authentic, `1` tampered
- `tamper_kind`: `D` splice, `S` copy-move, empty for authentic
- masks are `{0, 255}` (white = forged region)

Inspect a few samples in `ipynb/01_inspect_processed.ipynb`.

## Load in training code

```python
from src.data import CASIA2Dataset

train = CASIA2Dataset("data/processed/casia2", split="train")
batch = train[0]
# batch["image"]  float32 [3, 256, 256] in [0, 1]
# batch["mask"]   float32 [1, 256, 256] in {0, 1}
# batch["label"]  int64 0 or 1
# batch["stem"]   original filename stem
```
