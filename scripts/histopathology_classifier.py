"""Histopathology tile annotator for micro-sam.

Loads annotation tiles from h5 files produced by ``extract_ann_images.py``,
extracts the center 1/9 tile (ROI), computes UNI2 (histopathology foundation
model) features, trains a pixel classifier on user scribbles filtered by the
tissue mask, and saves predictions back to the same h5 file.

Usage::

    python scripts/histopathology_classifier.py /path/to/data.h5
"""

import os.path
import re
import subprocess
import tempfile
import time
from pathlib import Path

import h5py
import napari
import numpy as np
import timm
import torch
import torch.nn.functional as F
from bioimage_cpp.utils import Blocking
from matplotlib.colors import to_rgba
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.notifications import show_info
from qtpy import QtWidgets
from skimage.filters.rank import modal
from skimage.morphology import footprint_rectangle
from skimage.transform import resize as sk_resize
from torchvision import transforms
from tqdm import tqdm

from micro_sam import util
from micro_sam.pixel_classification import (
    _grid_shape,
    project_prediction_to_image,
    train_pixel_classifier,
)
from micro_sam.pixel_classification import (
    accumulate_pixel_labels as accumulate_pixel_labels_,
)
from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._annotator import _ClassifierBase
from micro_sam.sam_annotator._state import AnnotatorState

# ---------------------------------------------------------------------------
# Storage / sync configuration.
# ---------------------------------------------------------------------------
# Annotations are written to a small per-input sidecar ('<stem>.annot.h5'), never into the (large,
# reproducible) input h5 — so the input can live read-only on an external HDD while only the tiny
# sidecar is backed up. Keep the sidecar dir on the internal SSD (fast + safe). None -> next to input.
ANNOT_OUTPUT_DIR = "/Users/titus/growth_pattern_annotations"
# Dir for the large UNI2 feature cache (~up to 200 MB/tile). Never synced. None -> system temp.
# On the Transcend (1.5 TB free) next to the inputs so precomputed embeddings persist across reboots.
UNI_CACHE_DIR = "/Volumes/Transcend/growth_patterns/uni_cache"
# Shell script that pushes a finished sidecar to the HPC + HDD (edit its dest paths). Run per case.
SYNC_SCRIPT = Path(__file__).parent / "sync_case.sh"
# Folder of input h5s that `--precompute` (with no path arg) computes + caches embeddings for.
PRECOMPUTE_DIR = "/Volumes/Transcend/growth_patterns/annotation_h5"

# ---------------------------------------------------------------------------
# Class definitions – hardcoded as requested.
# ---------------------------------------------------------------------------
IGNORE_INDEX = 255  # 'excluded' tier — masked from ALL downstream use
TRUST_DEFAULT = 0
TRUST_REDUCED = (
    1  # low-confidence OR analysable-degraded (merged), on a real class
)

CLASS_IDS = {
    0: "background_glass",  # NON-TISSUE only: glass, empty space, alveolar air
    1: "stroma",  # reactive/desmoplastic stroma BETWEEN tumour structures
    2: "healthy_alveolar",
    3: "necrosis",
    4: "mucin_pool",  # route-out
    5: "mucinous_epithelium",  # route-out (cytology overrides architecture)
    6: "lepidic",  # ── growth patterns ──
    7: "acinar",
    8: "papillary",
    9: "micropapillary",  # overrides co-resident patterns within a shared airspace
    10: "complex_glandular",  # cribriform AND fused glands
    11: "solid",
    12: "cartilage",
    13: "blood_vessel",
    14: "bronchial_epithelium",
    15: "benign_glands",  # ACINAR mimic — verify, never merge into a pattern class
    16: "alveolar_macrophages",  # MICROPAPILLARY mimic — verify, never merge
    17: "tissue_other",  # FOREGROUND catch-all: pleura, free blood, pigment
}

PATTERN_CLASSES = (
    6,
    7,
    8,
    9,
    10,
    11,
)  # metadata for downstream consumers — not computed here
HIGH_GRADE = (9, 10, 11)
ROUTE_OUT = (4, 5)
BENIGN_STRUCT = (12, 13, 14, 15, 16)
NON_TISSUE = (0,)  # aligns with the foreground mask
TISSUE_CLASSES = tuple(c for c in CLASS_IDS if c != 0)

COLLAPSE_TO_TISSUE_OTHER = {12: 17, 13: 17}  # cartilage, vessel — safe
# 14, 15, 16 → verify against their mimicked pattern class before any collapse (C8)
# nothing ever collapses into 0 — background_glass is non-tissue only (C14)
# Fixed, high-contrast palette (Trubetskoy's "20 distinct colors") so every class id always renders
# in a well-separable color on the annotations/prediction layers instead of napari's hashed default.
# Colors are assigned to ids in sorted order, so the mapping stays stable if CLASS_IDS changes.
CLASS_PALETTE = {
    # ── non-tissue / neutral: desaturated, recedes ──
    0: "#ffffff",  # background_glass    — white = glass. Literal, and it disappears.
    17: "#c8c8c8",  # tissue_other        — neutral grey, clearly "tissue, unclassified"
    1: "#e0d4c8",  # stroma              — warm pale tan (abundant; must not shout)
    2: "#dfeaf2",  # healthy_alveolar    — pale blue-grey
    3: "#5a5a5a",  # necrosis            — dark grey
    # ── mucinous route-out: TEAL family (one hue, two lightnesses) ──
    4: "#7fd4cd",  # mucin_pool          — light teal
    5: "#128f86",  # mucinous_epithelium — dark teal
    # ── growth patterns: RED→ORANGE→YELLOW ramp, low→high grade ──
    6: "#fee08b",  # lepidic             — pale yellow  (low grade)
    7: "#fdae61",  # acinar              — orange
    8: "#f46d43",  # papillary           — deep orange
    9: "#d73027",  # micropapillary      — red         (high grade)
    10: "#a50026",  # complex_glandular   — dark red    (high grade)
    11: "#67001f",  # solid               — maroon      (high grade)
    # ── benign structures: PURPLE / BLUE family ──
    12: "#c6a5d8",  # cartilage           — light purple
    13: "#7b4fa3",  # blood_vessel        — purple
    14: "#4a6fd4",  # bronchial_epithelium— blue
    15: "#8c3f8c",  # benign_glands       — magenta-purple
    16: "#b07aa1",  # alveolar_macrophages— dusty mauve
}

IGNORE_COLOR = (
    "#000000"  # or render as transparent/hatched — NOT a class color
)


def _class_color(class_id):
    """Hex color pinned to a class id (by its sorted position in CLASS_IDS)."""
    order = sorted(CLASS_IDS)
    return CLASS_PALETTE[order.index(class_id) % len(CLASS_PALETTE)]


# Certainty labels painted on the "certainty" layer. Unpainted (0) = full certainty.
# Stored as raw ints; how 1/2 weight U-Net training is decided by the training consumer.
CERTAINTY_IDS = {1: "uncertain", 2: "excluded"}

# Side of the square neighbourhood for the majority (modal) smoothing of the prediction. Set to
# <= 1 to disable; larger removes more scatter but rounds off fine structures.
SMOOTHING_SIZE = 5


def _smooth_grid_prediction(pred, grid_shape):
    """Majority-filter a flat grid prediction to remove scattered single-cell misclassifications."""
    if SMOOTHING_SIZE <= 1 or len(grid_shape) != 2:
        return pred
    grid = np.asarray(pred).reshape(grid_shape).astype("uint8")
    smoothed = modal(
        grid, footprint_rectangle((SMOOTHING_SIZE, SMOOTHING_SIZE))
    )
    return smoothed.reshape(-1).astype(pred.dtype)


# h5 group holding precomputed UNI2 features, cached per image to avoid the ~20s recompute at the
# first RF training step. Entries are deleted once an image's prediction is saved.
UNI_CACHE_GROUP = "uni_features"


def _parse_bbox(name):
    """Extract the WSI bounding box (x, y, w, h) encoded in a tile filename, or None."""
    m = re.search(r"_x(\d+)_y(\d+)_w(\d+)_h(\d+)", name)
    if m is None:
        return None
    x, y, w, h = (int(v) for v in m.groups())
    return {"x": x, "y": y, "w": w, "h": h}


def _next_h5(path):
    """Return the next *.h5 file (sorted) in the same directory, or None if this is the last."""
    path = Path(path)
    siblings = sorted(path.parent.glob("*.h5"))
    names = [str(p) for p in siblings]
    if str(path) in names:
        idx = names.index(str(path))
        return siblings[idx + 1] if idx + 1 < len(siblings) else None
    return siblings[0] if siblings else None


# ---------------------------------------------------------------------------
# UNI2 feature extraction.
# ---------------------------------------------------------------------------
# The pixel classifier is fed features from UNI2-h (a histopathology foundation ViT), replacing
# SAM. UNI2 is run on the ROI at full resolution: the ROI is tiled with an overlap (halo) so
# border patches keep neighbouring-tissue context, each tile's coarse (patch/14) token grid is
# bilinearly upsampled onto a downsampled output feature grid, and the tiles are stitched.

UNI_MODEL_PATH = "/Users/titus/Desktop/go_annotation/univ2_model.bin"
UNI_PATCH = 14
UNI_TILE = (448, 448)  # inner tile size in ROI px (multiple of 14)
UNI_HALO = (
    56,
    56,
)  # overlap on each side in ROI px, context for border patches (multiple of 14)
# Output feature-grid longest side. UNI's genuine resolution is the patch stride (~1024/14 ≈ 73
# samples/side), so 256 already oversamples it while keeping RF predict fast and memory low
# (grid² × 1536 × 4 bytes; 256 ≈ 0.4 GB, 512 ≈ 1.6 GB and ~4× slower predict). Raise for smoother
# boundaries at the cost of speed/memory.
UNI_GRID_SIZE = 256
# Default PCA components for the classifier. RF predict on the 1536-d UNI features is dominated by
# memory bandwidth over the wide feature rows, so reducing to ~50 components speeds prediction ~2.4×
# with little accuracy loss. Exposed as the "top feature channels" control, so it stays adjustable.
DEFAULT_N_COMPONENTS = 50
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_uni_model_and_transform(device, model_path):
    model = timm.create_model(
        pretrained=False,
        model_name="vit_giant_patch14_224",
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model.to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    model.eval()
    return model, transform


def load_uni(device):
    """Load UNI2 and register a forward hook that captures the patch-token sequence.

    The hook on ``model.norm`` captures the post-norm tokens (identical to ``forward_features``'s
    return) into a holder dict, so feature extraction reads them without depending on the return.
    The ``transform`` (which resizes to 224) is intentionally dropped: we run UNI2 at full ROI
    resolution and normalise the crops ourselves.
    """
    model, _ = get_uni_model_and_transform(device, UNI_MODEL_PATH)
    holder = {}
    model.norm.register_forward_hook(
        lambda m, i, o: holder.__setitem__("tokens", o.detach())
    )
    return model, holder


def _round_to_patch(n):
    """Round a side length to the nearest positive multiple of the patch size."""
    return max(UNI_PATCH, int(round(n / UNI_PATCH)) * UNI_PATCH)


@torch.no_grad()
def _uni_patch_grid(model, holder, crop, device):
    """Run UNI2 on a full-resolution RGB crop and return its (C, Hp, Wp) patch-token grid."""
    rgb = util._to_image(
        crop
    )  # (H, W, 3) uint8, same channel mapping SAM used
    tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .to(device)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, -1, 1, 1)
    tensor = (tensor - mean) / std
    # UNI2's patch embedding requires each side to be divisible by the patch size, so resize the
    # crop to the nearest multiple of 14 (a ≤2% rescale that keeps the native magnification).
    h, w = tensor.shape[-2:]
    th, tw = _round_to_patch(h), _round_to_patch(w)
    if (th, tw) != (h, w):
        tensor = F.interpolate(
            tensor, size=(th, tw), mode="bilinear", align_corners=False
        )
    model.forward_features(tensor)  # fires the hook
    tokens = holder["tokens"]
    hp, wp = th // UNI_PATCH, tw // UNI_PATCH
    return (
        tokens[:, model.num_prefix_tokens :, :]
        .reshape(1, hp, wp, -1)
        .permute(0, 3, 1, 2)[0]
    )


@torch.no_grad()
def compute_uni_features(model, holder, image, device):
    """Compute per-pixel UNI2 features for an ROI, matching ``compute_pixel_features``'s contract.

    Returns the features flattened over the grid, of shape (grid_h * grid_w, C), and the grid shape.
    """
    height, width = image.shape[:2]
    grid, scale = _grid_shape((height, width), UNI_GRID_SIZE)
    tiling = Blocking([0, 0], [height, width], list(UNI_TILE))

    feature_image = None
    for block_id in range(tiling.number_of_blocks):
        block = tiling.get_block_with_halo(block_id, list(UNI_HALO))
        outer, inner_local = block.outer_block, block.inner_block_local
        crop = image[
            outer.begin[0] : outer.end[0], outer.begin[1] : outer.end[1]
        ]
        patch_grid = _uni_patch_grid(
            model, holder, crop, device
        )  # (C, Hp, Wp)
        channels, hp, wp = patch_grid.shape
        if feature_image is None:
            feature_image = np.zeros(grid + (channels,), dtype="float32")

        # Map the inner (non-halo) region into the tile's patch grid, dropping the overlap.
        outer_h, outer_w = (
            outer.end[0] - outer.begin[0],
            outer.end[1] - outer.begin[1],
        )
        sy, sx = hp / outer_h, wp / outer_w
        py0 = min(int(round(inner_local.begin[0] * sy)), hp - 1)
        px0 = min(int(round(inner_local.begin[1] * sx)), wp - 1)
        py1 = max(int(round(inner_local.end[0] * sy)), py0 + 1)
        px1 = max(int(round(inner_local.end[1] * sx)), px0 + 1)
        inner = patch_grid[:, py0:py1, px0:px1].unsqueeze(0)

        # Bilinearly upsample the inner patch grid onto its slot in the global output grid.
        gy0, gy1 = (
            outer.begin[0] + inner_local.begin[0],
            outer.begin[0] + inner_local.end[0],
        )
        gx0, gx1 = (
            outer.begin[1] + inner_local.begin[1],
            outer.begin[1] + inner_local.end[1],
        )
        by0, by1 = int(round(gy0 * scale)), int(round(gy1 * scale))
        bx0, bx1 = int(round(gx0 * scale)), int(round(gx1 * scale))
        if by1 - by0 < 1 or bx1 - bx0 < 1:
            continue
        upsampled = F.interpolate(
            inner,
            size=(by1 - by0, bx1 - bx0),
            mode="bilinear",
            align_corners=False,
        )
        feature_image[by0:by1, bx0:bx1] = (
            upsampled[0].permute(1, 2, 0).cpu().numpy()
        )

    return feature_image.reshape(-1, feature_image.shape[-1]), grid


def _add_roi_border(viewer, offset, roi_shape):
    """Draw a thin outline marking the ROI borders for the annotator."""
    y0, x0 = offset
    h, w = roi_shape
    rect = np.array([[y0, x0], [y0, x0 + w], [y0 + h, x0 + w], [y0 + h, x0]])
    if "roi border" in viewer.layers:
        del viewer.layers["roi border"]
    viewer.add_shapes(
        rect,
        shape_type="rectangle",
        name="roi border",
        edge_color="yellow",
        edge_width=6,
        face_color="transparent",
    )
    # Keep painting focus on the annotations layer, not the new shapes layer.
    if "annotations" in viewer.layers:
        viewer.layers.selection.active = viewer.layers["annotations"]


# ---------------------------------------------------------------------------
# H5 session helper
# ---------------------------------------------------------------------------


class HistopathologySession:
    """Loads one h5 file and manages per-image ROI extraction."""

    @staticmethod
    def get_center_roi(image, mask, max_h, max_w):
        """Return the center 1/9 tile from a padded image and mask."""
        y_start, y_end = max_h // 3, 2 * max_h // 3
        x_start, x_end = max_w // 3, 2 * max_w // 3
        return image[y_start:y_end, x_start:x_end], mask[
            y_start:y_end, x_start:x_end
        ]

    def __init__(self, h5_path):
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            self.n_images = f["images"].attrs["num_images"]
            self._image_shape = tuple(f["images"].shape[1:3])  # (max_h, max_w)
            self._attrs = dict(f.attrs)
            # Extract names from the filenames attribute.
            raw_names = f["images"].attrs["filenames"]
            # h5py returns names as a numpy ndarray (dtypes: object or bytes)
            if isinstance(raw_names, (list, tuple)):
                names = raw_names
            else:
                names = list(raw_names)
            self._names = [
                n.decode("utf-8")
                if isinstance(n, (bytes, bytearray))
                else str(n)
                for n in names
            ]
            if not self._names:
                self._names = [os.path.splitext(os.path.basename(h5_path))[0]]

        self.current_idx = 0
        self.roi_mask = None  # set when an image is loaded

        # Sidecar output (annotations only) and scratch feature cache — never written into the input.
        in_path = Path(self.h5_path)
        out_dir = (
            Path(ANNOT_OUTPUT_DIR) if ANNOT_OUTPUT_DIR else in_path.parent
        )
        self.out_path = str(out_dir / f"{in_path.stem}.annot.h5")
        cache_dir = (
            Path(UNI_CACHE_DIR)
            if UNI_CACHE_DIR
            else Path(tempfile.gettempdir())
        )
        self.cache_path = str(cache_dir / f"{in_path.stem}.unicache.h5")

    @property
    def image_shape(self):
        """(max_h, max_w) of the padded storage."""
        return self._image_shape

    @property
    def roi_shape(self):
        """Shape of the center 1/9 ROI."""
        h, w = self._image_shape
        return (h // 3, w // 3)

    @property
    def roi_offset(self):
        """(y_start, x_start) of the center 1/9 ROI within the padded image."""
        h, w = self._image_shape
        return (h // 3, w // 3)

    def get_current(self):
        """Return (full_image, image_roi, mask_roi) for the current index.

        The stored tissue mask is bit-packed along its width (``(H, W/8)`` uint8, 8 px per byte), so
        it is unpacked back to a full ``(H, W)`` 0/255 mask before the ROI is cropped. Any already
        full-width mask passes through and is just binarised to 0/255.
        """
        with h5py.File(self.h5_path, "r") as f:
            img = np.asarray(f["images"][self.current_idx])
            msk = np.asarray(f["masks"][self.current_idx])
        h, w = self._image_shape
        if msk.shape[-1] * 8 == w:  # bit-packed foreground mask -> unpack to (H, W)
            msk = np.unpackbits(msk, axis=-1)[..., :w]
        msk = np.where(msk > 0, 255, 0).astype("uint8")
        roi_img, roi_msk = self.get_center_roi(
            img, msk, self._image_shape[0], self._image_shape[1]
        )
        return img, roi_img, roi_msk

    def get_current_roi(self):
        """Return (image_roi, mask_roi) for the current index."""
        _, roi_img, roi_msk = self.get_current()
        return roi_img, roi_msk

    def save_predictions(self, predictions_dict):
        """Save annotation masks (+ reconstruction geometry) into the sidecar output h5.

        The input h5 is never written; only the small ``<stem>.annot.h5`` sidecar is, so the input can
        stay read-only and only a few-MB file needs backing up. Label maps are stored as gzipped uint8
        (class ids <= 15, certainty <= 2).

        Args:
            predictions_dict: image name -> {"prediction", "annotations", "certainty", "tissue_mask"}.
        """
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.out_path, "a") as f:
            preds_group = f.require_group("predictions")

            for name, data in predictions_dict.items():
                sub = preds_group.require_group(name)
                # Delete existing datasets so re-saving an edited image overwrites instead of raising
                # "name already exists"; 'certainty'/'tissue_mask' included so a cleared map drops.
                for key in (
                    "prediction",
                    "annotations",
                    "certainty",
                    "tissue_mask",
                ):
                    if key in sub:
                        del sub[key]
                sub.create_dataset(
                    "prediction",
                    data=np.asarray(data["prediction"], dtype="uint8"),
                    compression="gzip",
                )
                sub.create_dataset(
                    "annotations",
                    data=np.asarray(data["annotations"], dtype="uint8"),
                    compression="gzip",
                )
                # Only store the certainty map when something is painted (unpainted 0 = full certainty
                # everywhere, so an all-zero map is redundant). Raw uint8 ids 1/2.
                certainty = data.get("certainty")
                if certainty is not None and np.asarray(certainty).any():
                    sub.create_dataset(
                        "certainty",
                        data=np.asarray(certainty, dtype="uint8"),
                        compression="gzip",
                    )
                # The corrected foreground mask (was written back into the input 'masks' dataset).
                mask = data.get("tissue_mask")
                if mask is not None:
                    sub.create_dataset(
                        "tissue_mask",
                        data=np.asarray(mask, dtype="uint8"),
                        compression="gzip",
                    )
                # Geometry to place the ROI back into the WSI without re-parsing anything: the tile's
                # WSI bbox (from the filename) plus the ROI's offset/shape inside the padded tile.
                bbox = _parse_bbox(name)
                if bbox is not None:
                    sub.attrs.update(bbox)
                sub.attrs["roi_offset"] = list(self.roi_offset)
                sub.attrs["roi_shape"] = list(self.roi_shape)

            f.attrs["input_h5"] = os.path.basename(self.h5_path)
            f.attrs["image_shape"] = list(self._image_shape)
            f.attrs["class_ids"] = str(list(CLASS_IDS.items()))
            f.attrs["certainty_ids"] = str(list(CERTAINTY_IDS.items()))
            preds_group.attrs["n_saved"] = len(preds_group)

    def save_cached_features(self, name, features, grid_shape):
        """Cache UNI2 features for one image in the scratch cache file (float16, never synced)."""
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.cache_path, "a") as f:
            grp = f.require_group(UNI_CACHE_GROUP)
            if name in grp:
                del grp[name]
            ds = grp.create_dataset(
                name, data=np.asarray(features, dtype="float16")
            )
            ds.attrs["grid_shape"] = list(grid_shape)

    def load_cached_features(self, name):
        """Return cached (features float32, grid_shape) for an image, or None if not cached."""
        if not os.path.exists(self.cache_path):
            return None
        with h5py.File(self.cache_path, "r") as f:
            grp = f.get(UNI_CACHE_GROUP)
            if grp is None or name not in grp:
                return None
            ds = grp[name]
            return np.asarray(ds, dtype="float32"), tuple(
                int(v) for v in ds.attrs["grid_shape"]
            )

    def has_cached_features(self, name):
        """Whether complete features for ``name`` are cached — a cheap metadata-only check.

        Only reads the h5 directory/attrs, never the ~200 MB feature array (unlike
        ``load_cached_features``), so ``--precompute`` resume skips already-done tiles instantly
        instead of dragging every cached file back off the HDD. A partially written entry (dataset
        present but no ``grid_shape`` attr) or an unreadable/corrupt cache counts as "not cached",
        so it is recomputed rather than crashing a later load.
        """
        if not os.path.exists(self.cache_path):
            return False
        try:
            with h5py.File(self.cache_path, "r") as f:
                grp = f.get(UNI_CACHE_GROUP)
                if grp is None:
                    return False
                ds = grp.get(name)
                return ds is not None and "grid_shape" in ds.attrs
        except OSError:
            return False

    def load_prediction(self, name):
        """Return saved {prediction, annotations, certainty} for an image, or None if not saved."""
        if not os.path.exists(self.out_path):
            return None
        with h5py.File(self.out_path, "r") as f:
            grp = f.get("predictions")
            if grp is None or name not in grp:
                return None
            sub = grp[name]
            if "prediction" not in sub:
                return None
            return {
                "prediction": np.asarray(sub["prediction"]),
                "annotations": np.asarray(sub["annotations"])
                if "annotations" in sub
                else None,
                "certainty": np.asarray(sub["certainty"])
                if "certainty" in sub
                else None,
            }

    def load_saved_mask(self, name):
        """Return the corrected tissue mask (0/255 uint8) saved for an image, or None."""
        if not os.path.exists(self.out_path):
            return None
        with h5py.File(self.out_path, "r") as f:
            grp = f.get("predictions")
            if (
                grp is None
                or name not in grp
                or "tissue_mask" not in grp[name]
            ):
                return None
            return np.asarray(grp[name]["tissue_mask"], dtype="uint8")

    def saved_names(self):
        """Return the set of image names that already have a saved prediction."""
        if not os.path.exists(self.out_path):
            return set()
        with h5py.File(self.out_path, "r") as f:
            grp = f.get("predictions")
            return set(grp.keys()) if grp is not None else set()

    def has_prediction(self, name):
        """Whether a saved prediction exists for the given image name."""
        return name in self.saved_names()

    def n_saved(self):
        """Number of the session's images that have a saved prediction."""
        return len(self.saved_names() & set(self._names))

    def set_review(self, name, value):
        """Flag/unflag a tile for review in the sidecar (removable: ``value=False`` deletes the flag).

        Stored as an attribute on a dedicated ``review`` group, independent of whether the tile has
        a saved prediction, so it can be toggled at any time. Setting False when no sidecar exists is
        a no-op (nothing to clear).
        """
        if not value and not os.path.exists(self.out_path):
            return
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.out_path, "a") as f:
            grp = f.require_group("review")
            if value:
                grp.attrs[name] = True
            elif name in grp.attrs:
                del grp.attrs[name]

    def get_review(self, name):
        """Whether the given tile is currently flagged for review."""
        if not os.path.exists(self.out_path):
            return False
        with h5py.File(self.out_path, "r") as f:
            grp = f.get("review")
            return bool(
                grp is not None
                and name in grp.attrs
                and grp.attrs[name]
            )


# ---------------------------------------------------------------------------
# Annotator – subclass of _ClassifierBase with histopathology-specific hooks.
# ---------------------------------------------------------------------------


class HistopathologyAnnotator(_ClassifierBase):
    """GUI for histopathology tile annotation with SAM embeddings and pixel classifier."""

    rf_attr = "pixel_rf"
    features_attr = "pixel_features"
    aux_attr = "pixel_grid_shape"
    label_widget_title = "Pixel label names:"
    max_components = 256
    tool_key = "pixel"
    supports_apply_to_volume = False  # always full ROI, no slice concept

    # ----------------------------------------------------------
    # _ClassifierBase hooks.
    # ----------------------------------------------------------

    def _compute_features(self):
        state = AnnotatorState()
        if state.pixel_features is None:
            if state.image is None:
                return None, None
            # Prefer features precomputed and cached in the h5 file; otherwise compute now and cache
            # them, so a later revisit of this image is instant too.
            cached = (
                self._session.load_cached_features(state.image_name)
                if state.image_name
                else None
            )
            if cached is not None:
                features, grid_shape = cached
            else:
                features, grid_shape = compute_uni_features(
                    self._uni_model,
                    self._uni_holder,
                    state.image,
                    self._uni_device,
                )
                if state.image_name:
                    self._session.save_cached_features(
                        state.image_name, features, grid_shape
                    )
            state.pixel_features, state.pixel_grid_shape = features, grid_shape
        return state.pixel_features, state.pixel_grid_shape

    def _create_widgets(self):
        # UNI2 replaces SAM, so there is no embedding-precompute step; drop the SAM embedding widget
        # (base uses 'state.widgets.get("embeddings")', so removing it is safe).
        super()._create_widgets()
        self._widgets.pop("embeddings", None)

    def _compute_training_labels(self, aux):
        if "annotations" not in self._viewer.layers:
            return None
        state = AnnotatorState()
        grid_shape = state.pixel_grid_shape
        if grid_shape is None:
            return None
        return accumulate_pixel_labels_(
            self._viewer.layers["annotations"].data,
            grid_shape,
        )

    def _tissue_mask(self):
        """Return the current full-ROI foreground mask (0/255 uint8).

        Reads the editable 'tissue' layer when present (so manual include/exclude edits take
        effect), else falls back to the mask loaded from the h5 file.
        """
        if "tissue" in self._viewer.layers:
            return np.where(
                self._viewer.layers["tissue"].data > 0, 255, 0
            ).astype("uint8")
        return self._session.roi_mask

    def _train(
        self,
        features,
        labels,
        previous_features,
        previous_labels,
        n_components,
        random_state,
    ):
        # labels is the output of _compute_training_labels – a flat array
        # resized from the annotation layer down to the feature grid. The roi_mask
        # however is still at full ROI resolution so we have to downsample it
        # to the same grid shape first.
        state = AnnotatorState()
        grid_shape = state.pixel_grid_shape
        if grid_shape is None:
            return None
        roi_mask_grid = sk_resize(
            self._tissue_mask(),
            grid_shape,
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        ).astype(int)
        flat_mask = roi_mask_grid.reshape(-1)

        # Filter: only train on labeled pixels that are also tissue.
        valid = (labels != 0) & (flat_mask == 255)
        filtered_features = features[valid]
        filtered_labels = labels[valid]

        if len(filtered_labels) == 0:
            if previous_labels is None or len(previous_labels) == 0:
                widgets._generate_message(
                    "error",
                    "You have not provided any tissue annotations.",
                )
                return None
            # Fall back to previous features only (user is re-training from
            # a model loaded on a different image).
            filtered_features = previous_features
            filtered_labels = previous_labels

        return train_pixel_classifier(
            filtered_features,
            filtered_labels,
            previous_features=previous_features,
            previous_labels=previous_labels,
            n_components=n_components,
            random_state=random_state,
        )

    def _project_prediction(self, prediction, aux):
        roi_shape = self._session.roi_shape
        return project_prediction_to_image(prediction, aux, roi_shape)

    # ----------------------------------------------------------
    # Override: _predict_and_show – force non-tissue pixels to 0.
    # ----------------------------------------------------------

    def _predict_and_show(self, rf, features, aux, apply_to_volume=True):
        try:
            pred = rf.predict(features)
        except ValueError:
            return widgets._generate_message(
                "error",
                "The loaded classifier does not match the current embeddings. "
                "Recompute the embeddings with the restored settings before predicting.",
            )

        # Cheap majority (modal) smoothing at grid resolution to drop scattered single-cell
        # misclassifications. Done on the raw grid (the RF only outputs trained classes 2-6, never 0)
        # before projecting/masking, so it is ~65k px (sub-ms) and can't leak the non-tissue 0 label.
        pred = _smooth_grid_prediction(pred, aux)

        # Project the grid-resolution prediction up to the full ROI shape first, then force
        # non-tissue pixels to 0. The RandomForest output ('pred') is at feature-grid resolution
        # while 'roi_mask' is at full ROI resolution, so the masking must happen after projection.
        prediction = self._project_prediction(pred, aux)
        if prediction is None:
            return None
        prediction[self._tissue_mask() == 0] = 0
        layer = self._viewer.layers["prediction"]
        layer.data = prediction
        self._refresh_label_widget()

    # ----------------------------------------------------------
    # Override: _update_image – work in ROI coordinate space.
    # ----------------------------------------------------------

    def _apply_class_colormap(self):
        """Pin each class id to its fixed CLASS_PALETTE color on the annotations/prediction layers.

        napari keeps a set colormap across '.data' resets, so this only needs to run when the layers
        are (re)created; '_update_image' is the funnel all load paths route through.
        """
        color_dict = {cid: to_rgba(_class_color(cid)) for cid in CLASS_IDS}
        color_dict[0] = (0.0, 0.0, 0.0, 0.0)  # background: transparent
        color_dict[None] = (0.5, 0.5, 0.5, 1.0)  # fallback for any unlisted id
        cmap = DirectLabelColormap(color_dict=color_dict)
        for name in ("annotations", "prediction"):
            if name in self._viewer.layers:
                self._viewer.layers[name].colormap = cmap

    def _update_image(self, segmentation_result=None):
        state = AnnotatorState()
        if state.skip_recomputing_embeddings:
            return
        if state.image_shape is None:
            return

        roi_shape = self._session.roi_shape
        self._ndim = 2
        self._shape = roi_shape

        if self._apply_to_volume is not None:
            self._apply_to_volume.visible = False

        self._invalidate_features()
        self._require_layers()
        self._apply_class_colormap()
        scale = (
            None
            if state.image_scale is None
            else tuple(state.image_scale)[: self._ndim]
        )

        self._viewer.layers["annotations"].data = np.zeros(
            roi_shape, dtype="uint32"
        )
        self._viewer.layers["prediction"].data = np.zeros(
            roi_shape, dtype="uint32"
        )
        # The certainty layer marks regions of uncertain (1) or excluded (2) labels for the U-Net
        # training consumer; base '_require_layers' only makes annotations/prediction, so create it
        # here. Unpainted (0) = full certainty, so an untouched layer needs no storage (see _do_save).
        if "certainty" not in self._viewer.layers:
            certainty_layer = self._viewer.add_labels(
                np.zeros(roi_shape, dtype="uint8"), name="certainty"
            )
            # Match the annotations brush so painting is usable on large ROIs (napari's default is
            # a few px); the base sets a proportional brush there in __init__.
            if "annotations" in self._viewer.layers:
                certainty_layer.brush_size = self._viewer.layers[
                    "annotations"
                ].brush_size
        else:
            self._viewer.layers["certainty"].data = np.zeros(
                roi_shape, dtype="uint8"
            )
        # The tissue layer exposes the (editable) foreground mask as binary 1/0: paint to include a
        # region, erase to exclude it. It is the source of truth for _tissue_mask(); UNI2 features
        # cover the whole ROI, so including a previously-masked region needs no recompute.
        tissue = (self._session.roi_mask == 255).astype("uint8")
        if "tissue" not in self._viewer.layers:
            tissue_layer = self._viewer.add_labels(tissue, name="tissue")
            tissue_layer.selected_label = 1
            if "annotations" in self._viewer.layers:
                tissue_layer.brush_size = self._viewer.layers[
                    "annotations"
                ].brush_size
        else:
            self._viewer.layers["tissue"].data = tissue
        # The label layers are ROI-sized; translate them to the ROI position so scribbles and
        # predictions overlay the center of the full "context" image displayed behind them.
        offset = self._session.roi_offset
        for name in ("annotations", "prediction", "certainty", "tissue"):
            self._viewer.layers[name].translate = offset
            if scale is not None:
                self._viewer.layers[name].scale = scale

        self._reorder_layers()

    def _reorder_layers(self):
        """Enforce a stable stacking so switching images doesn't bury the label layers.

        Re-adding 'context'/'image' on load pushes them to the top of the stack, hiding the
        tissue mask and prediction; this restores a fixed bottom→top order (labels on top) and
        keeps 'annotations' the active layer for painting.
        """
        order = [
            "context",
            "image",
            "roi border",
            "tissue",
            "certainty",
            "prediction",
            "annotations",
        ]
        for name in order:
            if name in self._viewer.layers:
                idx = self._viewer.layers.index(name)
                self._viewer.layers.move(idx, len(self._viewer.layers))
        if "annotations" in self._viewer.layers:
            self._viewer.layers.selection.active = self._viewer.layers[
                "annotations"
            ]

    # ----------------------------------------------------------
    # Widgets
    # ----------------------------------------------------------

    def _create_class_id_widget(self):
        """Create a grid of buttons for class ID selection."""
        group = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("Label Buttons")
        title.setToolTip("Click a button to set the brush label")
        layout.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(4)

        row, col = 0, 0
        for class_id, class_name in sorted(CLASS_IDS.items()):
            btn = QtWidgets.QPushButton(f"ID {class_id}: {class_name}")
            btn.setAutoExclusive(False)

            # Read from CLASS_PALETTE directly (not the layer) so buttons match the fixed layer
            # colors even though they are built before the first _apply_class_colormap.
            color = _class_color(class_id)
            btn.setStyleSheet(
                f"background-color: {color}; border: 1px solid #888; "
                f"padding: 4px; text-align: left;"
            )
            btn.setToolTip(f"Set brush label to ID {class_id} ({class_name})")
            btn.clicked.connect(
                lambda checked, cid=class_id: self._set_label(cid)
            )
            grid.addWidget(btn, row, col)

            col += 1
            if col > 2:
                col = 0
                row += 1

        layout.addLayout(grid)
        group.setLayout(layout)
        return group

    def _set_label(self, label_id):
        # Apply to both label layers so the same buttons work whether the user is scribbling on
        # 'annotations' or hand-correcting the 'prediction' layer.
        for name in ("annotations", "prediction"):
            if name in self._viewer.layers:
                self._viewer.layers[name].selected_label = label_id
                self._viewer.layers[name].mode = "paint"

    def _create_certainty_id_widget(self):
        """Create a separate row of buttons for painting the certainty layer."""
        group = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("Certainty Buttons")
        title.setToolTip(
            "Mark regions where the label is uncertain or should be excluded"
        )
        layout.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(4)
        for col, (cid, cname) in enumerate(sorted(CERTAINTY_IDS.items())):
            btn = QtWidgets.QPushButton(f"{cid}: {cname}")
            btn.setToolTip(
                f"Paint certainty {cid} ({cname}) on the certainty layer"
            )
            btn.clicked.connect(
                lambda checked, c=cid: self._set_certainty_label(c)
            )
            grid.addWidget(btn, 0, col)

        layout.addLayout(grid)
        group.setLayout(layout)
        return group

    def _set_certainty_label(self, label_id):
        if "certainty" in self._viewer.layers:
            layer = self._viewer.layers["certainty"]
            layer.selected_label = label_id
            layer.mode = "paint"
            self._viewer.layers.selection.active = layer

    def _create_erase_non_tissue_widget(self):
        """Create the erase non-tissue button."""
        btn = QtWidgets.QPushButton("Erase Non-Tissue [E]")
        btn.setToolTip("Clear all annotations in non-tissue areas of the ROI")

        @self._viewer.bind_key("e", overwrite=True)
        def _erase(event=None):
            self._erase_non_tissue()

        btn.clicked.connect(self._erase_non_tissue)
        return btn

    def _erase_non_tissue(self):
        """Set both the annotation and prediction layers to 0 where the tissue mask is 0."""
        roi_mask = self._tissue_mask()
        cleared = False
        for name in ("annotations", "prediction"):
            if name not in self._viewer.layers:
                continue
            layer = self._viewer.layers[name]
            if roi_mask.shape != layer.data.shape:
                widgets._generate_message(
                    "error",
                    f"Tissue mask shape does not match the {name} layer.",
                )
                return
            layer.data[roi_mask == 0] = 0
            layer.refresh()
            cleared = True
        if not cleared:
            widgets._generate_message(
                "error", "No annotation or prediction layer found."
            )

    def _create_previous_image_widget(self):
        """Create the previous image button."""
        btn = QtWidgets.QPushButton("Previous")
        btn.setToolTip("Load the previous image tile from the h5 file")
        btn.clicked.connect(self._previous_image)
        return btn

    def _create_next_image_widget(self):
        """Create the next image button."""
        btn = QtWidgets.QPushButton(
            f"Next [{self._session.current_idx + 1}/{self._session.n_images}]"
        )
        btn.setToolTip("Load the next image tile from the h5 file")
        btn.clicked.connect(self._next_image)
        self._next_button = btn
        return btn

    def _step_image(self, delta):
        """Move `delta` images through the session (wrapping) and load the result."""
        state = AnnotatorState()
        state.skip_recomputing_embeddings = False
        state.pixel_features = None
        self._invalidate_features()

        self._session.current_idx = (
            self._session.current_idx + delta
        ) % self._session.n_images
        self._load_current_image()

    def _next_image(self):
        """Advance to the next image in the session."""
        self._step_image(1)

    def _previous_image(self):
        """Go back to the previous image in the session."""
        self._step_image(-1)

    def _load_current_image(self):
        """Load the current ROI and set up napari layers."""
        session = self._session
        full_image, image, mask = session.get_current()
        # Prefer a previously saved corrected mask (input is read-only, so edits live in the sidecar).
        saved_mask = session.load_saved_mask(
            session._names[session.current_idx]
        )
        session.roi_mask = (
            saved_mask
            if saved_mask is not None and saved_mask.shape == mask.shape
            else mask
        )

        state = AnnotatorState()
        roi_shape = session.roi_shape
        offset = session.roi_offset
        state.image_shape = roi_shape
        state.ndim = 2
        state.image_name = session._names[session.current_idx]

        # Display the whole padded image as context, and overlay the ROI (which drives embeddings
        # and prediction) at its center via 'translate'.
        viewer = self._viewer
        for name in ("context", "image"):
            if name in viewer.layers:
                del viewer.layers[name]
        viewer.add_image(full_image, name="context", rgb=True)
        viewer.add_image(image, name="image", rgb=True, translate=offset)
        _add_roi_border(viewer, offset, roi_shape)

        state.image_shape = roi_shape
        state.image = image
        state.image_scale = viewer.layers["image"].scale

        # Invalidate the cached UNI2 features so the next 'Train and Predict' recomputes them for
        # this ROI instead of reusing the previous image's.
        state.pixel_features = None
        state.skip_recomputing_embeddings = False
        self._invalidate_features()

        # Reset annotation layers.
        self._update_image()

        # Update button text.
        self._next_button.setText(
            f"Next [{session.current_idx + 1}/{session.n_images}]"
        )

        # Refresh label widget, the saved/progress indicator, and this tile's review flag.
        self._refresh_label_widget()
        self._update_status()
        self._refresh_review_checkbox()

    def _create_review_widget(self):
        """Checkbox flagging the current tile for review; persisted per-tile in the sidecar."""
        box = QtWidgets.QCheckBox("Mark for review")
        box.setToolTip(
            "Flag this tile for later review. Stored per tile in the sidecar and restored when you "
            "revisit it; uncheck to remove."
        )
        box.toggled.connect(self._set_review)
        self._review_checkbox = box
        return box

    def _set_review(self, checked):
        """Persist the review flag for the current tile (called on user toggle)."""
        state = AnnotatorState()
        name = state.image_name or str(self._session.current_idx)
        self._session.set_review(name, bool(checked))

    def _refresh_review_checkbox(self):
        """Reflect the current tile's stored review flag without firing _set_review."""
        box = getattr(self, "_review_checkbox", None)
        if box is None:
            return
        state = AnnotatorState()
        name = state.image_name or str(self._session.current_idx)
        box.blockSignals(True)
        box.setChecked(self._session.get_review(name))
        box.blockSignals(False)

    def _create_load_prediction_widget(self):
        """Button to load a previously saved prediction for this image (hidden if none exists)."""
        btn = QtWidgets.QPushButton("Load Saved Prediction")
        btn.setToolTip(
            "Load the prediction/annotations saved earlier for this image from the h5 file"
        )
        btn.clicked.connect(self._load_prediction)
        self._load_button = btn
        return btn

    def _load_prediction(self):
        """Restore the saved prediction/annotations/certainty for the current image."""
        state = AnnotatorState()
        name = state.image_name or str(self._session.current_idx)
        data = self._session.load_prediction(name)
        if data is None:
            widgets._generate_message(
                "error", f"No saved prediction found for '{name}'."
            )
            return
        self._require_layers()
        self._viewer.layers["prediction"].data = data["prediction"].astype(
            "uint32"
        )
        if data["annotations"] is not None:
            self._viewer.layers["annotations"].data = data[
                "annotations"
            ].astype("uint32")
        if (
            data["certainty"] is not None
            and "certainty" in self._viewer.layers
        ):
            self._viewer.layers["certainty"].data = data["certainty"].astype(
                "uint8"
            )
        self._reorder_layers()
        self._refresh_label_widget()
        show_info(f"Loaded saved prediction for '{name}'.")

    def _create_save_widget(self):
        """Create the save prediction button."""
        btn = QtWidgets.QPushButton("Save [S]")
        btn.setToolTip(
            "Save the current prediction and annotations "
            "to the h5 file under predictions/<image_name>/"
        )

        @self._viewer.bind_key("s", overwrite=True)
        def _save(event=None):
            self._do_save()

        btn.clicked.connect(self._do_save)
        return btn

    def _do_save(self):
        """Save prediction and annotations for the current image."""
        state = AnnotatorState()
        if "prediction" not in self._viewer.layers:
            widgets._generate_message(
                "error", "No prediction layer found. Train and predict first."
            )
            return
        if "annotations" not in self._viewer.layers:
            widgets._generate_message("error", "No annotations layer found.")
            return

        # Re-mask the prediction with the (possibly edited) tissue mask so post-predict include/
        # exclude edits are reflected and only foreground pixels end up in the saved label.
        mask = self._tissue_mask()
        pred = self._viewer.layers["prediction"].data.copy()
        pred[mask == 0] = 0
        ann = self._viewer.layers["annotations"].data
        certainty = (
            self._viewer.layers["certainty"].data
            if "certainty" in self._viewer.layers
            else None
        )
        name = state.image_name or str(self._session.current_idx)

        # Keep the corrected foreground mask in memory; it is persisted in the sidecar below.
        self._session.roi_mask = mask

        # Build predictions dict for batch saving (all into the sidecar output h5).
        self._session.save_predictions(
            {
                name: {
                    "prediction": pred,
                    "annotations": ann,
                    "certainty": certainty,
                    "tissue_mask": mask,
                },
            }
        )
        # Keep the cached UNI2 features (they live on the roomy HDD cache, not the input h5), so
        # revisiting/re-annotating this image later is instant instead of a ~20s recompute.
        self._update_status()
        show_info(
            f"Saved prediction and annotations for '{name}' to "
            f"{self._session.h5_path}"
        )

    # ----------------------------------------------------------
    # _ClassifierBase widget override – insert new widgets.
    # ----------------------------------------------------------

    def _extra_classification_sections(self):
        """Return widgets shown above the settings dropdown."""
        widget_list = [
            self._create_status_widget(),
            self._create_class_id_widget(),
            self._create_certainty_id_widget(),
            self._create_erase_non_tissue_widget(),
            self._create_previous_image_widget(),
            self._create_next_image_widget(),
            self._create_review_widget(),
            self._create_load_prediction_widget(),
            self._create_save_widget(),
            self._create_close_series_widget(),
        ]
        return widget_list

    # ----------------------------------------------------------
    # Status indicator and series handling.
    # ----------------------------------------------------------

    def _create_status_widget(self):
        """Label showing whether the current image is saved and the series progress."""
        label = QtWidgets.QLabel()
        label.setWordWrap(True)
        self._status_label = label
        self._update_status()
        return label

    def _update_status(self):
        """Refresh the saved/progress indicator; turns green once the whole series is saved."""
        label = getattr(self, "_status_label", None)
        if label is None:
            return
        session = self._session
        name = session._names[session.current_idx]
        n_saved, total = session.n_saved(), session.n_images
        done = n_saved >= total
        current = "✓ saved" if session.has_prediction(name) else "— not saved"
        all_done = "  ✓ ALL SAVED" if done else ""
        label.setText(
            f"Current image: {current}\nSeries: {n_saved}/{total} saved{all_done}"
        )
        label.setStyleSheet(
            "padding:4px; border:1px solid #888;"
            + (" background-color:#2a7d4f; color:white;" if done else "")
        )
        # Only show "Load Saved Prediction" when this image actually has one cached.
        load_btn = getattr(self, "_load_button", None)
        if load_btn is not None:
            load_btn.setVisible(session.has_prediction(name))

    def _sync_case(self, out_path):
        """Push a finished sidecar to HPC + HDD in the background via sync_case.sh (fire-and-forget).

        The sidecar is a few MB, so this rarely takes long, but running it detached keeps napari
        responsive and avoids Qt-thread issues. Output is captured to a per-case log next to the
        sidecar; the script self-verifies the HPC copy and skips gracefully if dests are unset.
        """
        if not os.path.exists(out_path):
            return  # nothing was saved for this case
        log_dir = Path(out_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"sync_{Path(out_path).stem}_{ts}.log"
        log = open(log_path, "w")
        subprocess.Popen(
            ["bash", str(SYNC_SCRIPT), out_path],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        show_info(
            f"Syncing {Path(out_path).name} in background — log: {log_path}"
        )

    def _create_close_series_widget(self):
        """Button to delete this h5's cached embeddings and move on to the next h5 file."""
        btn = QtWidgets.QPushButton("Close Series")
        btn.setToolTip(
            "Delete the cached UNI2 embeddings for this h5 file and load the next h5 in the folder"
        )
        btn.clicked.connect(self._close_series)
        return btn

    def _close_series(self):
        """Confirm, delete cached embeddings, and load the next h5 file (or report completion)."""
        session = self._session
        n_saved, total = session.n_saved(), session.n_images
        reply = QtWidgets.QMessageBox.question(
            self,
            "Close series",
            f"Close this series? {n_saved}/{total} images saved.",
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        # Keep the cached UNI2 embeddings (stored on the HDD) so this series stays instant to reopen.
        # Back up the finished case's sidecar (HPC + HDD) before moving on — including the last case.
        self._sync_case(session.out_path)
        next_path = _next_h5(session.h5_path)
        if next_path is None:
            show_info(
                "All series completed – no further h5 files to annotate."
            )
            return

        self._session = HistopathologySession(next_path)
        self._session.current_idx = 0
        state = AnnotatorState()
        state.pixel_features = None
        self._invalidate_features()
        self._load_current_image()
        show_info(f"Loaded next series: {Path(next_path).name}")

    # ----------------------------------------------------------
    # init – inject session and load first image.
    # ----------------------------------------------------------

    def __init__(self, viewer, session):
        """Create the annotator.

        Args:
            viewer: The napari viewer.
            session: A :class:HistopathologySession instance.
        """
        # Set _session before calling super().__init__(), which calls
        # _create_widgets() -> _extra_classification_sections() that need
        # self._session to be available.
        self._session = session
        self._next_button = None
        self._load_button = None
        self._review_checkbox = None
        super().__init__(viewer)

        # Load UNI2 once; its features drive the pixel classifier (computed lazily in
        # _compute_features and cached on the state per image).
        self._uni_device = util.get_device()
        self._uni_model, self._uni_holder = load_uni(self._uni_device)

        # Enable PCA (top feature channels) by default to keep RF prediction fast on the 1536-d UNI
        # features; still adjustable via the "top feature channels" control.
        self._set_options(True, DEFAULT_N_COMPONENTS, False)

        # The base _ClassifierBase creates the annotation layer with a tiny
        # default brush_size that is unusable for large ROIs (1024⁺ px).
        # Set it proportional to the canvas size so the user can actually see
        # where they are painting without zooming in.
        if "annotations" in self._viewer.layers:
            shape = self._viewer.layers["annotations"].data.shape
            min_side = min(shape)
            # 1% of the canvas, clamped to sensible bounds
            self._viewer.layers["annotations"].brush_size = max(
                5, min(int(min_side * 0.01), 100)
            )

        # Now that all widgets exist, set the first image's load-button visibility and review flag.
        self._update_status()
        self._refresh_review_checkbox()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def precompute_session_features(
    session, model, holder, device, position=1, leave=False
):
    """Compute and cache UNI2 features for every ROI in one session's h5 file.

    Shows a per-sample tqdm bar at ``position`` with ``leave=False`` so that, when driving a whole
    folder, the bar is cleared and reused for the next file (replaced, not stacked) beneath the
    global file bar. ``model``/``holder``/``device`` are passed in so UNI2 is loaded only once.
    """
    original_idx = session.current_idx
    bar = tqdm(
        range(session.n_images),
        position=position,
        leave=leave,
        desc=Path(session.h5_path).stem,
        unit="img",
    )
    for idx in bar:
        name = session._names[idx]
        if session.has_cached_features(name):
            bar.set_postfix_str("cached")
            continue
        session.current_idx = idx
        roi_img, _ = session.get_current_roi()
        features, grid_shape = compute_uni_features(
            model, holder, roi_img, device
        )
        session.save_cached_features(name, features, grid_shape)
        bar.set_postfix_str("computed")
    bar.close()
    session.current_idx = original_idx


def precompute_folder(folder):
    """Precompute + cache UNI2 features for every input ``*.h5`` in ``folder``.

    A global tqdm bar tracks files (position 0); each file's per-sample bar (position 1) is replaced
    as the run advances. UNI2 is loaded once and reused across all files.
    """
    folder = Path(folder)
    files = sorted(
        p
        for p in folder.glob("*.h5")
        if not p.name.endswith((".annot.h5", ".unicache.h5"))
    )
    if not files:
        raise SystemExit(f"No input .h5 files found in {folder}")
    if UNI_CACHE_DIR is None:
        print(
            "WARNING: UNI_CACHE_DIR is None → embeddings cache to the system temp dir and may be "
            "cleared on reboot. Set UNI_CACHE_DIR to a persistent folder to keep them."
        )
    device = util.get_device()
    model, holder = load_uni(device)
    for path in tqdm(files, position=0, desc="h5 files", unit="file"):
        precompute_session_features(
            HistopathologySession(path), model, holder, device, position=1
        )


def main(h5_path=None):
    if h5_path is None:
        import sys

        if len(sys.argv) > 1:
            h5_path = sys.argv[1]
        else:
            from qtpy.QtWidgets import QFileDialog

            path, _ = QFileDialog.getOpenFileName(
                None,
                "Select h5 file",
                "",
                "HDF5 Files (*.h5 *.hdf5)",
            )
            if not path:
                raise SystemExit("No h5 file selected.")
            h5_path = path

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise SystemExit(f"h5 file not found: {h5_path}")

    # Load session.
    session = HistopathologySession(h5_path)

    # Start napari viewer.
    viewer = napari.Viewer()

    # Create and inject the first image: the whole padded image as context, and the ROI (which
    # drives embeddings and prediction) overlaid at its center via 'translate'.
    full_image, image, mask = session.get_current()
    saved_mask = session.load_saved_mask(session._names[0])
    session.roi_mask = (
        saved_mask
        if saved_mask is not None and saved_mask.shape == mask.shape
        else mask
    )
    viewer.add_image(full_image, name="context", rgb=True)
    viewer.add_image(
        image, name="image", rgb=True, translate=session.roi_offset
    )
    _add_roi_border(viewer, session.roi_offset, session.roi_shape)

    # Set up state.
    state = AnnotatorState()
    state.image_shape = session.roi_shape
    state.ndim = 2
    state.image_name = session._names[0]
    state.image = image
    state.image_scale = viewer.layers["image"].scale

    # Create annotator (loads UNI2, which drives the pixel classifier).
    annotator = HistopathologyAnnotator(viewer, session)
    annotator._update_image()

    # Add annotator dock widget.
    viewer.window.add_dock_widget(
        annotator,
        name="(Histopathology) Pixel Classifier",
    )

    napari.run()


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if "--precompute" in args:
        # Precompute + cache UNI2 features, then exit (no GUI). With no path arg, process every h5 in
        # PRECOMPUTE_DIR; a path arg may be a folder (all its h5s) or a single h5 file.
        args = [a for a in args if a != "--precompute"]
        target = Path(args[0]) if args else Path(PRECOMPUTE_DIR)
        if not target.exists():
            raise SystemExit(f"path not found: {target}")
        try:
            if target.is_dir():
                precompute_folder(target)
            else:
                device = util.get_device()
                model, holder = load_uni(device)
                precompute_session_features(
                    HistopathologySession(target),
                    model,
                    holder,
                    device,
                    position=0,
                    leave=True,
                )
        except KeyboardInterrupt:
            # Each finished tile is flushed to its own cache file, so whatever completed is kept;
            # rerun the same command to resume from the first uncached tile.
            print(
                "\nInterrupted — cached features so far are saved; rerun to resume."
            )
            raise SystemExit(130)
    else:
        main(args[0] if args else None)
