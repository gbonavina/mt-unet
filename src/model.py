"""MT-UNet (Zhu, Rohling & Salcudean, arXiv:2202.07118).

Matches the authors' code (github.com/hz-zhu/MT-UNet): U-Net with classification
from concatenated global-average-pooled bottleneck and decoder-top features,
nearest upsample (not transposed conv), and the Eq. 5 uncertainty-weighted loss.

CASIA2 uses RGB input and a binary tamper mask, so the dense head is trained
with BCE-with-logits instead of spatial-softmax KLD (the paper's saliency maps
sum to 1; authentic CASIA2 masks are all zeros and are not a distribution).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 conv + BatchNorm + ReLU blocks (paper Fig. 1)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def up_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Nearest upsample then 3x3 conv + BatchNorm + ReLU (paper: Upsampling)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def classification_head(
    in_features: int,
    mid_features: int,
    out_features: int,
    dropout_rate: float = 0.25,
) -> nn.Sequential:
    """Linear → dropout 25% → ReLU → linear (authors' classification_head)."""
    return nn.Sequential(
        nn.Linear(in_features, mid_features),
        nn.Dropout(p=dropout_rate),
        nn.ReLU(inplace=True),
        nn.Linear(mid_features, out_features),
    )


class MT_UNET(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512, 1024),
    ) -> None:
        super().__init__()
        if len(features) != 5:
            raise ValueError("MT-UNet uses five widths: 64, 128, 256, 512, 1024")

        f = features
        self.conv1 = conv_block(in_channels, f[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(f[0], f[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv_block(f[1], f[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = conv_block(f[2], f[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = conv_block(f[3], f[4])

        self.up5 = up_conv(f[4], f[3])
        self.up_conv5 = conv_block(f[4], f[3])
        self.up4 = up_conv(f[3], f[2])
        self.up_conv4 = conv_block(f[3], f[2])
        self.up3 = up_conv(f[2], f[1])
        self.up_conv3 = conv_block(f[2], f[1])
        self.up2 = up_conv(f[1], f[0])
        self.up_conv2 = conv_block(f[1], f[0])

        # GAP(bottleneck) ⊕ GAP(decoder top) → two linear layers, dropout 0.25
        self.out_classification = classification_head(
            in_features=f[0] + f[4],
            mid_features=f[0],
            out_features=num_classes,
            dropout_rate=0.25,
        )
        self.out_conv_image = nn.Sequential(
            nn.Conv2d(f[0], f[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(f[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(f[0], out_channels, kernel_size=1, stride=1, padding=0),
        )

        # σ_c starts at 1. Eq. 5 keeps σ_s fixed at 1 (only class uncertainty is learned).
        self.lg_sigma_class = nn.Parameter(torch.zeros(1))

    def _align(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="nearest")
        return up

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool1(e1))
        e3 = self.conv3(self.pool2(e2))
        e4 = self.conv4(self.pool3(e3))
        e5 = self.conv5(self.pool4(e4))

        d5 = self.up_conv5(torch.cat((e4, self._align(self.up5(e5), e4)), dim=1))
        d4 = self.up_conv4(torch.cat((e3, self._align(self.up4(d5), e3)), dim=1))
        d3 = self.up_conv3(torch.cat((e2, self._align(self.up3(d4), e2)), dim=1))
        d2 = self.up_conv2(torch.cat((e1, self._align(self.up2(d3), e1)), dim=1))

        pooled = torch.cat((e5.mean(dim=(-2, -1)), d2.mean(dim=(-2, -1))), dim=1)
        y_class = self.out_classification(pooled)
        y_mask = self.out_conv_image(d2)
        return y_mask, y_class

    def compute_loss(
        self,
        y_mask_pred: torch.Tensor,
        y_class_pred: torch.Tensor,
        y_mask_true: torch.Tensor,
        y_class_true: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Paper Eq. 5 (MTLS3): L = L_s + L_c / σ_c² + ln(σ_c + 1).

        L_c is cross-entropy on class logits. L_s is BCE-with-logits on the
        binary tamper mask (CASIA2 stand-in for the paper's KLD saliency loss).
        """
        sigma_c = torch.exp(self.lg_sigma_class)
        class_loss_raw = F.cross_entropy(y_class_pred, y_class_true)
        mask_loss_raw = F.binary_cross_entropy_with_logits(y_mask_pred, y_mask_true)
        class_loss_weighted = class_loss_raw / (sigma_c**2) + torch.log(sigma_c + 1.0)
        return {
            "loss_sum": mask_loss_raw + class_loss_weighted,
            "class_loss_raw": class_loss_raw,
            "mask_loss_raw": mask_loss_raw,
            "sigma_c": sigma_c.detach(),
        }

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sigmoid mask probabilities and softmax class probabilities."""
        mask_logits, class_logits = self.forward(x)
        return torch.sigmoid(mask_logits), torch.softmax(class_logits, dim=1)

    def fit(
        self,
        train_data: Dataset | DataLoader,
        val_data: Dataset | DataLoader | None = None,
        *,
        epochs: int = 50,
        batch_size: int = 8,
        lr: float = 1e-4,
        lr_factor: float = 0.1,
        lr_patience: int = 10,
        lr_min: float = 1e-8,
        early_stop_patience: int | None = None,
        num_workers: int = 0,
        device: str | torch.device | None = None,
        checkpoint_path: str | Path | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train with Adam and ReduceLROnPlateau (paper Appendix B, Eq. 5 loss).

        ``train_data`` / ``val_data`` may be a Dataset or a DataLoader. When the
        validation loss plateaus, the scheduler cuts the learning rate and the
        best weights are reloaded, matching the paper's RLRP scheme.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        if early_stop_patience is None:
            early_stop_patience = lr_patience * 2 + 3

        train_loader = _as_loader(train_data, batch_size, shuffle=True, num_workers=num_workers)
        val_loader = (
            None
            if val_data is None
            else _as_loader(val_data, batch_size, shuffle=False, num_workers=num_workers)
        )

        self.to(device)
        optimizer = Adam(self.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_cls": [],
            "train_mask": [],
            "val_cls": [],
            "val_mask": [],
            "lr": [],
            "sigma_c": [],
        }
        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        best_epoch = 0
        checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

        for epoch in range(1, epochs + 1):
            train_stats = self._run_epoch(train_loader, device, optimizer, verbose, f"train {epoch}/{epochs}")
            if val_loader is not None:
                val_stats = self._run_epoch(val_loader, device, None, verbose, f"val {epoch}/{epochs}")
                monitor = val_stats["loss_sum"]
            else:
                val_stats = {key: float("nan") for key in train_stats}
                monitor = train_stats["loss_sum"]

            history["train_loss"].append(train_stats["loss_sum"])
            history["train_cls"].append(train_stats["class_loss_raw"])
            history["train_mask"].append(train_stats["mask_loss_raw"])
            history["val_loss"].append(val_stats["loss_sum"])
            history["val_cls"].append(val_stats["class_loss_raw"])
            history["val_mask"].append(val_stats["mask_loss_raw"])
            history["sigma_c"].append(train_stats["sigma_c"])

            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(monitor)
            new_lr = optimizer.param_groups[0]["lr"]
            history["lr"].append(new_lr)

            improved = monitor < best_val
            if improved:
                best_val = monitor
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, checkpoint_path)

            if new_lr < prev_lr and best_state is not None:
                self.load_state_dict(best_state)
                if verbose:
                    print(f"- lr reduced to {new_lr:.2e}; restored best epoch {best_epoch}")

            if verbose:
                val_msg = (
                    f" val {val_stats['loss_sum']:.4f}"
                    if val_loader is not None
                    else ""
                )
                print(
                    f"epoch {epoch:4d}  train {train_stats['loss_sum']:.4f}"
                    f"{val_msg}  cls {train_stats['class_loss_raw']:.4f}"
                    f"  mask {train_stats['mask_loss_raw']:.4f}"
                    f"  σc {train_stats['sigma_c']:.3f}  lr {new_lr:.2e}"
                )

            if new_lr < lr_min:
                if verbose:
                    print(f"- early stop: lr {new_lr:.2e} below {lr_min:.2e}")
                break
            if epoch - best_epoch >= early_stop_patience:
                if verbose:
                    print(f"- early stop: no improvement for {early_stop_patience} epochs")
                break

        if best_state is not None:
            self.load_state_dict(best_state)
        return history

    def _run_epoch(
        self,
        loader: DataLoader,
        device: torch.device,
        optimizer: Adam | None,
        verbose: bool,
        desc: str,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.train(training)
        totals = {"loss_sum": 0.0, "class_loss_raw": 0.0, "mask_loss_raw": 0.0}
        seen = 0
        sigma_c = float(torch.exp(self.lg_sigma_class).detach().cpu())
        batches = tqdm(loader, desc=desc, leave=False) if verbose else loader
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for batch in batches:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                mask_pred, class_pred = self(images)
                losses = self.compute_loss(mask_pred, class_pred, masks, labels)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    losses["loss_sum"].backward()
                    optimizer.step()
                batch_n = images.size(0)
                seen += batch_n
                for key in totals:
                    totals[key] += float(losses[key].detach()) * batch_n
                sigma_c = float(losses["sigma_c"].detach().cpu())
        means = {key: value / max(seen, 1) for key, value in totals.items()}
        means["sigma_c"] = sigma_c
        return means


def _as_loader(
    data: Dataset | DataLoader,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    if isinstance(data, DataLoader):
        return data
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def test() -> None:
    x = torch.randn(2, 3, 256, 256)
    model = MT_UNET()
    mask_logits, class_logits = model(x)
    assert mask_logits.shape == (2, 1, 256, 256)
    assert class_logits.shape == (2, 2)

    mask_true = torch.zeros(2, 1, 256, 256)
    mask_true[1, :, 40:80, 40:80] = 1.0
    class_true = torch.tensor([0, 1], dtype=torch.long)
    losses = model.compute_loss(mask_logits, class_logits, mask_true, class_true)
    losses["loss_sum"].backward()
    print("mask", tuple(mask_logits.shape), "class", tuple(class_logits.shape))
    print("loss", float(losses["loss_sum"].detach()))


if __name__ == "__main__":
    test()
