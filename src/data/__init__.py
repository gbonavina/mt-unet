"""CASIA2 data loading."""

from src.data.dataset import CASIA2Dataset
from src.data.pairing import Sample, pair_casia2

__all__ = ["CASIA2Dataset", "Sample", "pair_casia2"]
