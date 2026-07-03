"""Non-GUI tests for histopathology classifier core logic."""

from pathlib import Path

import h5py
import numpy as np

from scripts.histopathology_classifier import (
    CLASS_IDS,
    HistopathologySession,
)

TEST_H5 = Path("/tmp/test_histopath_classifier.h5")


def create_test_h5(path: Path, n_images: int = 3, max_h: int = 300, max_w: int = 400):
    """Create a test h5 file with synthetic images and tissue masks."""
    images = np.random.randint(50, 200, (n_images, max_h, max_w, 3), dtype=np.uint8)
    masks = np.zeros((n_images, max_h, max_w), dtype=np.uint8)

    for i in range(n_images):
        # Center ROI is tissue
        y0, y1 = max_h // 3, 2 * max_h // 3
        x0, x1 = max_w // 3, 2 * max_w // 3
        masks[i, y0:y1, x0:x1] = 255
        # Random blobs outside center
        for _ in range(3):
            cy, cx = np.random.randint(max_h), np.random.randint(max_w)
            masks[i, cy - 20:cy + 20, cx - 20:cx + 20] = 255

    filenames = [f"img_{i:03d}" for i in range(n_images)]
    with h5py.File(path, "w") as f:
        imgs_ds = f.create_dataset("images", data=images)
        imgs_ds.attrs["num_images"] = n_images
        imgs_ds.attrs["filenames"] = [n.encode("utf-8") for n in filenames]
        msk_ds = f.create_dataset("masks", data=masks)
        msk_ds.attrs["num_images"] = n_images
        f.attrs["split"] = "test"
        f.attrs["subtype"] = "tumor"
        f.attrs["entity"] = "new_tumor"
        f.attrs["wsi_path"] = "/dev/null"


def test_session_loads():
    """Session opens the h5 and reads metadata."""
    create_test_h5(TEST_H5)
    sess = HistopathologySession(TEST_H5)
    assert sess.h5_path == str(TEST_H5)
    assert sess.n_images == 3
    assert sess.image_shape == (300, 400)
    assert sess.roi_shape == (100, 133)  # 300//3, 400//3
    assert sess.roi_offset == (100, 133)  # 300//3, 400//3
    assert sess._names == ["img_000", "img_001", "img_002"]
    print("PASS: test_session_loads")


def test_get_current_full_and_roi():
    """get_current returns the full padded image plus the center ROI crop."""
    create_test_h5(TEST_H5, max_h=300, max_w=400)
    sess = HistopathologySession(TEST_H5)
    full, roi_img, roi_msk = sess.get_current()
    assert full.shape == (300, 400, 3)   # whole padded image (context)
    assert roi_img.shape == (100, 133, 3)
    assert roi_msk.shape == (100, 133)
    print("PASS: test_get_current_full_and_roi")


def test_h5_names_numpy_array():
    """h5py returns filenames as numpy ndarray, not list."""
    create_test_h5(TEST_H5)
    with h5py.File(TEST_H5, "r") as f:
        raw = f["images"].attrs["filenames"]
        assert isinstance(raw, np.ndarray), f"Expected ndarray, got {type(raw)}"
    sess = HistopathologySession(TEST_H5)
    assert len(sess._names) == 3


def test_roi_extracts_center():
    """ROI is the exact center 1/9 tile."""
    # Use known dimensions where 1/3 division is exact
    create_test_h5(TEST_H5, max_h=300, max_w=400)
    sess = HistopathologySession(TEST_H5)
    img, msk = sess.get_current_roi()
    assert img.shape == (100, 133, 3)
    assert msk.shape == (100, 133)


def test_roi_mask_is_tissue():
    """The center ROI mask should have 255s (tissue)."""
    create_test_h5(TEST_H5, max_h=300, max_w=400)
    sess = HistopathologySession(TEST_H5)
    img, msk = sess.get_current_roi()
    assert msk.sum() > 0, "Center ROI should be tissue (mask > 0)"


def test_all_images_accessible():
    """Can iterate through all images."""
    create_test_h5(TEST_H5, n_images=5, max_h=300, max_w=400)
    sess = HistopathologySession(TEST_H5)
    assert sess.n_images == 5
    for i in range(5):
        sess.current_idx = i
        img, msk = sess.get_current_roi()
        assert img.shape == (100, 133, 3)
    print("PASS: test_all_images_accessible")


def test_save_predictions():
    """Saving predictions writes to h5 under the predictions/ group."""
    # Create fresh h5 for first save
    path1 = TEST_H5.parent / "test_save1.h5"
    create_test_h5(path1, max_h=300, max_w=400)
    sess = HistopathologySession(path1)
    roi_shape = sess.roi_shape

    pred = np.random.randint(0, 6, roi_shape, dtype="int32")
    ann = np.random.randint(0, 6, roi_shape, dtype="int32")
    sess.save_predictions({
        "img_000": {"prediction": pred, "annotations": ann},
    })

    with h5py.File(path1, "r") as f:
        assert "predictions" in f
        assert "img_000" in f["predictions"]
        assert np.array_equal(f["predictions"]["img_000/prediction"][()], pred)
        assert np.array_equal(f["predictions"]["img_000/annotations"][()], ann)
        assert f["predictions"].attrs["n_saved"] == 1

    # Re-saving the same image (after editing) must overwrite, not crash.
    pred2 = np.random.randint(0, 6, roi_shape, dtype="int32")
    sess.save_predictions({"img_000": {"prediction": pred2, "annotations": ann}})
    with h5py.File(path1, "r") as f:
        assert np.array_equal(f["predictions"]["img_000/prediction"][()], pred2)
        assert f["predictions"].attrs["n_saved"] == 1  # still one image, not duplicated

    # Save a second image to a different file to verify accumulation
    path2 = TEST_H5.parent / "test_save2.h5"
    create_test_h5(path2, max_h=300, max_w=400)
    sess2 = HistopathologySession(path2)
    sess2.current_idx = 0
    sess2.save_predictions({"img_000": {"prediction": pred, "annotations": ann}})
    sess2.current_idx = 1
    sess2.save_predictions({"img_001": {"prediction": pred, "annotations": ann}})
    with h5py.File(path2, "r") as f:
        assert "img_000" in f["predictions"]
        assert "img_001" in f["predictions"]
        assert f["predictions"].attrs["n_saved"] == 2

    path1.unlink()
    path2.unlink()
    print("PASS: test_save_predictions")


def test_class_ids():
    """CLASS_IDS contains the expected keys and values."""
    assert CLASS_IDS == {
        2: "tumor",
        3: "stroma",
        4: "necrosis",
        5: "lymphocyte",
        6: "background",
    }
    print("PASS: test_class_ids")


# ---------------------------------------------------------------------------
# Tissue mask filtering logic (standalone test — no GUI needed).
# ---------------------------------------------------------------------------

def test_tissue_mask_filtering():
    """Simulates the tissue mask filtering done in _train()."""
    roi_shape = (10, 5)
    roi_mask = np.zeros(roi_shape, dtype=np.uint8)
    roi_mask[:5, :] = 255  # top 5 rows = tissue (25 total pixels)

    flat_labels = np.array([
        2, 3, 0, 4, 0,
        2, 0, 3, 0, 0,
        0, 2, 0, 0, 1,
        0, 0, 0, 0, 0,
        0, 3, 0, 4, 0,
        2, 0, 0, 3, 0,
        0, 0, 0, 0, 0,
        4, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ], dtype=np.int64)

    assert len(flat_labels) == 50
    flat_mask = roi_mask.reshape(-1)
    assert len(flat_mask) == 50
    assert flat_mask[:25].sum() == 25 * 255
    assert flat_mask[25:].sum() == 0

    valid = (flat_labels != 0) & (flat_mask == 255)

    expected = 9
    assert valid.sum() == expected, f"Expected {expected}, got {valid.sum()}"

    non_tissue_labeled = valid & (flat_mask == 0)
    assert non_tissue_labeled.sum() == 0
    print("PASS: test_tissue_mask_filtering")


def test_prediction_force_non_tissue_zero():
    """Simulates prediction forcing non-tissue pixels to 0."""
    # Alternating tissue/non-tissue: positions 0,2,4,6,8 are tissue (255)
    flat_mask = np.array([255, 0, 255, 0, 255, 0, 255, 0, 255, 0] * 10)
    pred = np.array([2, 3, 4, 2, 2, 3, 3, 4, 4, 2] * 10)

    # Force non-tissue to 0 (mask == 0 means background)
    pred_before = pred.copy()
    pred[flat_mask == 0] = 0

    assert (pred[flat_mask == 0] == 0).all(), "Non-tissue pixels should be 0"
    # Ensure we actually changed something
    non_tissue_set_zero = (pred == 0) & (pred_before != 0)
    assert non_tissue_set_zero.sum() == 50, "Should have zeroed 50 non-tissue pixels"
    print("PASS: test_prediction_force_non_tissue_zero")


def test_train_mask_alignment_to_grid_shape():
    """Ensure roi_mask is downsampled to grid shape before filtering labels.

    The annotation layer is at full ROI resolution (e.g. 1024×1024) but labels
    from ``accumulate_pixel_labels`` are already resized down to the feature
    grid (e.g. 256×256 = 65536 rows).  The mask must be resized to match
    grid_shape, otherwise boolean indexing fails.
    """
    import numpy as np
    from skimage.transform import resize as sk_resize

    # Simulate: 1024×1024 ROI with 8192×256 feature matrix
    # Grid shape is 256×256 = 65536
    grid_shape = (256, 256)
    roi_mask_2d = np.zeros((1024, 1024), dtype=np.uint8)
    # Top-left quadrant is tissue
    roi_mask_2d[:512, :512] = 255

    # Resize mask to grid shape (this is what the fixed _train does)
    roi_mask_grid = sk_resize(
        roi_mask_2d, grid_shape,
        order=0, anti_aliasing=False, preserve_range=True,
    ).astype(int)
    flat_mask = roi_mask_grid.reshape(-1)

    # Simulate labels (grid size = 65536)
    n_rows = grid_shape[0] * grid_shape[1]
    labels = np.zeros(n_rows, dtype=np.int64)
    # Put some labels in grid area [32:64, 32:64] which maps to tissue
    for r in range(32, 64):
        for c in range(32, 64):
            idx = r * grid_shape[1] + c
            labels[idx] = 2

    valid = (labels != 0) & (flat_mask == 255)
    tissue_count = valid.sum()
    assert tissue_count == 1024, \
        f"Should have 1024 tissue labels, got {tissue_count}"

    # Add labels to non-tissue quadrant [192:224, 192:224]
    for r in range(192, 224):
        for c in range(192, 224):
            idx = r * grid_shape[1] + c
            labels[idx] = 3

    valid_after = (labels != 0) & (flat_mask == 255)
    # Those 1024 non-tissue labels should be filtered out so total stays 1024
    assert valid_after.sum() == 1024, \
        f"Non-tissue labels should be filtered (expected 1024, got {valid_after.sum()})"

    # Verify the non-tissue labels ARE in the array but were excluded
    labeled_but_non_tissue = (labels != 0) & (flat_mask == 0)
    assert labeled_but_non_tissue.sum() == 1024, \
        f"Should have 1024 non-tissue labels (expected 1024, got {labeled_but_non_tissue.sum()})"
    print("PASS: test_train_mask_alignment_to_grid_shape")


if __name__ == "__main__":
    test_session_loads()
    test_get_current_full_and_roi()
    test_h5_names_numpy_array()
    test_roi_extracts_center()
    test_roi_mask_is_tissue()
    test_all_images_accessible()
    test_save_predictions()
    test_class_ids()
    test_tissue_mask_filtering()
    test_prediction_force_non_tissue_zero()
    test_train_mask_alignment_to_grid_shape()
    print("\nAll tests passed!")
