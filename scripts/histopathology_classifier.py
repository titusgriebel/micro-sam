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
from pathlib import Path

import h5py
import napari
import numpy as np
import timm
import torch
import torch.nn.functional as F
from bioimage_cpp.utils import Blocking
from napari.utils.notifications import show_info
from qtpy import QtWidgets
from skimage.filters.rank import modal
from skimage.morphology import footprint_rectangle
from skimage.transform import resize as sk_resize
from torchvision import transforms

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
# Class definitions – hardcoded as requested.
# ---------------------------------------------------------------------------

CLASS_IDS = {
    2: "tumor",
    3: "stroma",
    4: "necrosis",
    5: "lymphocyte",
    6: "background",
}

# Certainty labels painted on the "certainty" layer. Unpainted (0) = full certainty.
# Stored as raw ints; how 1/2 weight U-Net training is decided by the training consumer.
CERTAINTY_IDS = {1: "uncertain", 2: "excluded"}

# Side of the square neighbourhood for the majority (modal) smoothing of the prediction. Set to
# <= 1 to disable; larger removes more scatter but rounds off fine structures.
SMOOTHING_SIZE = 3


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
        """Return (full_image, image_roi, mask_roi) for the current index."""
        with h5py.File(self.h5_path, "r") as f:
            img = np.asarray(f["images"][self.current_idx])
            msk = np.asarray(f["masks"][self.current_idx])
        roi_img, roi_msk = self.get_center_roi(
            img, msk, self._image_shape[0], self._image_shape[1]
        )
        return img, roi_img, roi_msk

    def get_current_roi(self):
        """Return (image_roi, mask_roi) for the current index."""
        _, roi_img, roi_msk = self.get_current()
        return roi_img, roi_msk

    def save_roi_mask(self, idx, roi_mask):
        """Overwrite the center 1/9 of the stored padded mask with the corrected ROI mask.

        Uses the same center slices as ``get_center_roi`` and keeps the stored 0/255 dtype.
        """
        h, w = self._image_shape
        y0, y1 = h // 3, 2 * h // 3
        x0, x1 = w // 3, 2 * w // 3
        with h5py.File(self.h5_path, "a") as f:
            f["masks"][idx, y0:y1, x0:x1] = roi_mask.astype(f["masks"].dtype)

    def save_predictions(self, predictions_dict):
        """Save prediction and annotation arrays to the h5 file.

        Args:
            predictions_dict: dict mapping image name ->
                {"prediction": np.ndarray, "annotations": np.ndarray}
        """
        with h5py.File(self.h5_path, "a") as f:
            preds_group = f.require_group("predictions")

            for name, data in predictions_dict.items():
                sub = preds_group.require_group(name)
                # Delete existing datasets so re-saving an edited image overwrites instead of
                # raising "name already exists". 'certainty' is included so re-saving an image
                # whose certainty was cleared drops the stale dataset (it is only recreated below
                # when non-empty).
                for key in ("prediction", "annotations", "certainty"):
                    if key in sub:
                        del sub[key]
                sub.create_dataset(
                    "prediction",
                    data=data["prediction"],
                    dtype="int32",
                )
                sub.create_dataset(
                    "annotations",
                    data=data["annotations"],
                    dtype="int32",
                )
                # Only store the certainty map when something is painted (unpainted 0 = full
                # certainty everywhere, so an all-zero map is redundant). Raw uint8 ids 1/2.
                certainty = data.get("certainty")
                if certainty is not None and np.asarray(certainty).any():
                    sub.create_dataset(
                        "certainty", data=certainty, dtype="uint8"
                    )
                # Store the ROI's WSI bounding box (parsed from the tile filename) so the prediction
                # can be placed back into the whole-slide image without re-parsing the name.
                bbox = _parse_bbox(name)
                if bbox is not None:
                    sub.attrs.update(bbox)

            preds_group.attrs["h5_path"] = self.h5_path
            preds_group.attrs["class_ids"] = str(list(CLASS_IDS.items()))
            preds_group.attrs["certainty_ids"] = str(
                list(CERTAINTY_IDS.items())
            )
            # Count actual saved images, not input dict size
            preds_group.attrs["n_saved"] = len(preds_group)

    def save_cached_features(self, name, features, grid_shape):
        """Cache UNI2 features for one image in the h5 file (float16 to halve the footprint)."""
        with h5py.File(self.h5_path, "a") as f:
            grp = f.require_group(UNI_CACHE_GROUP)
            if name in grp:
                del grp[name]
            ds = grp.create_dataset(
                name, data=np.asarray(features, dtype="float16")
            )
            ds.attrs["grid_shape"] = list(grid_shape)

    def load_cached_features(self, name):
        """Return cached (features float32, grid_shape) for an image, or None if not cached."""
        with h5py.File(self.h5_path, "r") as f:
            grp = f.get(UNI_CACHE_GROUP)
            if grp is None or name not in grp:
                return None
            ds = grp[name]
            return np.asarray(ds, dtype="float32"), tuple(
                int(v) for v in ds.attrs["grid_shape"]
            )

    def delete_cached_features(self, name):
        """Drop an image's cached features (called once its prediction has been saved)."""
        with h5py.File(self.h5_path, "a") as f:
            grp = f.get(UNI_CACHE_GROUP)
            if grp is not None and name in grp:
                del grp[name]

    def delete_all_cached_features(self):
        """Drop every cached feature array (called when closing the series)."""
        with h5py.File(self.h5_path, "a") as f:
            if UNI_CACHE_GROUP in f:
                del f[UNI_CACHE_GROUP]

    def load_prediction(self, name):
        """Return saved {prediction, annotations, certainty} for an image, or None if not saved."""
        with h5py.File(self.h5_path, "r") as f:
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

    def saved_names(self):
        """Return the set of image names that already have a saved prediction."""
        with h5py.File(self.h5_path, "r") as f:
            grp = f.get("predictions")
            return set(grp.keys()) if grp is not None else set()

    def has_prediction(self, name):
        """Whether a saved prediction exists for the given image name."""
        return name in self.saved_names()

    def n_saved(self):
        """Number of the session's images that have a saved prediction."""
        return len(self.saved_names() & set(self._names))


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

            color = "#cccccc"
            try:
                ann_layer = self._viewer.layers["annotations"]
                c = ann_layer.get_color(class_id)
                r, g, b = (int(round(255 * ch)) for ch in c[:3])
                color = f"rgb({r}, {g}, {b})"
            except (KeyError, IndexError, AttributeError):
                # Annotations layer may not exist during widget creation.
                pass
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
        session.roi_mask = mask

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

        # Refresh label widget and the saved/progress indicator.
        self._refresh_label_widget()
        self._update_status()

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
        if data["certainty"] is not None and "certainty" in self._viewer.layers:
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

        # Persist the corrected foreground mask: overwrite the stored mask and keep it in memory.
        self._session.roi_mask = mask
        self._session.save_roi_mask(self._session.current_idx, mask)

        # Build predictions dict for batch saving.
        self._session.save_predictions(
            {
                name: {
                    "prediction": pred,
                    "annotations": ann,
                    "certainty": certainty,
                },
            }
        )
        # The annotation for this image is done, so drop its cached features to reclaim h5 space.
        # The in-memory features (state.pixel_features) stay until Next, so re-training is instant.
        self._session.delete_cached_features(name)
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
            f"Close this series? {n_saved}/{total} images saved.\n"
            "The cached UNI2 embeddings for this h5 file will be deleted.",
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        session.delete_all_cached_features()
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

        # Now that all widgets exist, set the first image's load-button visibility.
        self._update_status()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def precompute_features(session, verbose=True):
    """Compute and cache UNI2 features for every ROI in the session's h5 file.

    Run this before annotating (``--precompute``) so the first RF training on each image loads
    features from the h5 cache instead of waiting for the ~20s UNI2 forward pass. Cached entries are
    removed automatically once an image's prediction is saved.
    """
    device = util.get_device()
    model, holder = load_uni(device)
    original_idx = session.current_idx
    for idx in range(session.n_images):
        name = session._names[idx]
        if session.load_cached_features(name) is not None:
            if verbose:
                print(f"[{idx + 1}/{session.n_images}] already cached: {name}")
            continue
        session.current_idx = idx
        roi_img, _ = session.get_current_roi()
        features, grid_shape = compute_uni_features(
            model, holder, roi_img, device
        )
        session.save_cached_features(name, features, grid_shape)
        if verbose:
            print(f"[{idx + 1}/{session.n_images}] cached: {name}")
    session.current_idx = original_idx


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
    session.roi_mask = mask
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
        # Precompute and cache UNI2 features for the whole h5 file, then exit (no GUI).
        args = [a for a in args if a != "--precompute"]
        if not args:
            raise SystemExit(
                "Usage: histopathology_classifier.py --precompute <file.h5>"
            )
        h5 = Path(args[0])
        if not h5.exists():
            raise SystemExit(f"h5 file not found: {h5}")
        precompute_features(HistopathologySession(h5))
    else:
        main(args[0] if args else None)
