"""
data.py

Datasets, preprocessing, and line segmentation utilities for printed OCR.
Assumes you already converted PDFs to page PNGs and created a CSV index
with columns: page_image_path, text (either page- or line-level).
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


# ---------- basic image preprocessing ----------

def preprocess_page(path: Path, deskew: bool = True) -> np.ndarray:
    """
    Load a page image, convert to grayscale, binarize with Otsu, optionally deskew.

    Returns:
        bin_img: binary image (uint8) with text as black (0) and background as white (255)
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu binarization
    _, bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if not deskew:
        return bin_img

    # Deskew using minimum area rectangle on text pixels
    coords = np.column_stack(np.where(bin_img < 255))
    if coords.size == 0:
        return bin_img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = bin_img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    bin_img = cv2.warpAffine(
        bin_img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return bin_img


def crop_main_text(bin_img: np.ndarray, margin: int = 10) -> np.ndarray:
    """
    Heuristic: crop the densest block of text to ignore decorations/marginalia.
    Works on binarized image (0 text, 255 background).
    """
    text_mask = (bin_img == 0).astype(np.float32)
    rows = text_mask.mean(axis=1)
    cols = text_mask.mean(axis=0)

    row_thresh = 0.02
    col_thresh = 0.02

    row_indices = np.where(rows > row_thresh)[0]
    col_indices = np.where(cols > col_thresh)[0]

    # Fallback if no text is detected
    if len(row_indices) == 0 or len(col_indices) == 0:
        return bin_img

    r1, r2 = row_indices[0], row_indices[-1]
    c1, c2 = col_indices[0], col_indices[-1]

    r1 = max(r1 - margin, 0)
    c1 = max(c1 - margin, 0)
    r2 = min(r2 + margin, bin_img.shape[0] - 1)
    c2 = min(c2 + margin, bin_img.shape[1] - 1)

    return bin_img[r1:r2 + 1, c1:c2 + 1]


# ---------- line segmentation ----------

def segment_lines(bin_img: np.ndarray,
                  min_line_height: int = 10) -> List[np.ndarray]:
    """
    Segment a binarized page into line images using horizontal projections.

    Returns:
        list of binary line images (cropped by rows).
    """
    text_mask = (bin_img == 0).astype(np.float32)
    row_hist = text_mask.mean(axis=1)
    thresh = 0.01

    lines: List[Tuple[int, int]] = []
    in_line = False
    start = 0

    for i, val in enumerate(row_hist):
        if val > thresh and not in_line:
            in_line = True
            start = i
        elif val <= thresh and in_line:
            in_line = False
            if i - start >= min_line_height:
                lines.append((start, i))

    if in_line:
        lines.append((start, len(row_hist)))

    crops = [bin_img[s:e, :] for s, e in lines]
    return crops


# ---------- vocabulary and text encoding ----------

def build_vocab(texts: List[str]) -> Tuple[dict, dict]:
    """
    Build char-level vocabulary from a list of strings.
    Returns:
        char2idx, idx2char dicts. Index 0 reserved for CTC 'blank'.
    """
    chars = set()
    for t in texts:
        chars.update(list(t))

    chars = sorted(chars)
    char2idx = {c: i + 1 for i, c in enumerate(chars)}  # start at 1
    char2idx["<blank>"] = 0

    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def encode_text(text: str, char2idx: dict) -> List[int]:
    """Convert text string to list of int indices using char2idx."""
    return [char2idx[c] for c in text if c in char2idx]


# ---------- PyTorch dataset ----------

class LineOCRDataset(Dataset):
    """
    Dataset for line-level OCR.

    Expects a CSV with columns:
        page_image_path: path to line image
        text:      ground-truth transcription for that line
    """

    def __init__(self,
                 csv_path: str,
                 char2idx: dict,
                 img_height: int = 32):
        self.df = pd.read_csv(csv_path)
        self.char2idx = char2idx
        self.img_height = img_height

    def __len__(self):
        return len(self.df)

    def _load_and_process_image(self, img_path: str) -> np.ndarray:
        """
        Load a line image (already segmented) and resize to fixed height,
        keep aspect ratio, then normalize to [0, 1].
        """
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")

        h, w = img.shape
        scale = self.img_height / h
        new_w = int(w * scale)

        img = cv2.resize(img, (new_w, self.img_height),
                         interpolation=cv2.INTER_AREA)

        img = img.astype("float32") / 255.0
        img = 1.0 - img  # make text bright if you prefer
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["page_image_path"]
        text = str(row["text"])

        img = self._load_and_process_image(img_path)  # (H, W)
        label = encode_text(text, self.char2idx)

        # Convert to tensors
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(label, dtype=torch.long)

        return {
            "image": img,
            "label": label,
            "label_len": torch.tensor(len(label), dtype=torch.long),
            "text": text,
            "path": img_path,
        }


def collate_fn(batch):
    """
    Collate function for DataLoader:
    - pads images in width to the max width in batch
    - concatenates labels and records lengths
    """
    images = [b["image"] for b in batch]
    labels = [b["label"] for b in batch]
    label_lens = [b["label_len"] for b in batch]

    max_w = max(img.shape[-1] for img in images)
    h = images[0].shape[-2]

    # Pad images to same width
    padded = []
    for img in images:
        _, _, w = img.shape
        pad_w = max_w - w
        if pad_w > 0:
            pad = torch.zeros((1, h, pad_w), dtype=img.dtype)
            img = torch.cat([img, pad], dim=-1)
        padded.append(img)

    images = torch.stack(padded, dim=0)  # (N, 1, H, W)

    labels_cat = torch.cat(labels)
    label_lens = torch.stack(label_lens)

    return {
        "images": images,
        "labels": labels_cat,
        "label_lens": label_lens,
        "texts": [b["text"] for b in batch],
        "paths": [b["path"] for b in batch],
    }
