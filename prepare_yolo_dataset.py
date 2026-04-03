import os
import random
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from scipy import ndimage
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
RASTER_FILE  = 'input_raster.tif'
MASK_FILE    = 'sam2_segmentation_mask.tif'
VECTOR_FILE  = 'points.geojson'   # or .shp, .gpkg
OUTPUT_DIR   = 'yolo_dataset'

WINDOW_SIZE  = 1024
CLASS_ID     = 0      # single class
VAL_SPLIT    = 0.2    # 20% validation
RANDOM_SEED  = 42


def get_pixel_coords(gdf, transform, width, height):
    """Convert all GeoDataFrame point geometries to pixel (col, row) tuples."""
    coords = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            coords.append(None)
            continue
        col, row_img = ~transform * (geom.x, geom.y)
        px, py = int(col), int(row_img)
        if 0 <= px < width and 0 <= py < height:
            coords.append((px, py))
        else:
            coords.append(None)   # outside raster bounds
    return coords


def normalize_to_uint8(img_patch):
    """Normalize any dtype image patch to uint8 for saving."""
    if img_patch.dtype == np.uint8:
        return img_patch
    img_min = img_patch.min()
    img_max = img_patch.max()
    denom = float(img_max - img_min) if img_max > img_min else 1.0
    img_patch = ((img_patch.astype(np.float32) - img_min) / denom * 255.0)
    return np.clip(img_patch, 0, 255).astype(np.uint8)


def get_yolo_boxes_in_window(mask_patch, points_in_window, win_w, win_h):
    """
    For each point inside the window, find its connected component in the mask
    and return a unique YOLO-format bounding box string.

    Returns a deduplicated list of label strings:
        "class_id cx cy w h"  (all values normalized 0-1)
    """
    binary = (mask_patch > 0).astype(np.uint8)
    labeled, _ = ndimage.label(binary)

    seen_components = set()
    labels = []

    for (pt_idx, x_local, y_local) in points_in_window:
        comp_id = labeled[y_local, x_local]
        if comp_id == 0:
            print(f"      [!] Point {pt_idx + 1} is on mask background. Skipping.")
            continue
        if comp_id in seen_components:
            continue   # same object already added by another nearby point
        seen_components.add(comp_id)

        ys, xs = np.where(labeled == comp_id)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Reject if the bounding box touches any edge of the window —
        # that means the object is cut off and the bbox would be incomplete.
        if x_min == 0 or y_min == 0 or x_max == win_w - 1 or y_max == win_h - 1:
            print(f"      [!] Point {pt_idx + 1} bbox touches window edge — object cut off. Skipping.")
            seen_components.discard(comp_id)  # allow re-evaluation from another window
            continue

        # YOLO: center x, center y, width, height — all normalized
        cx = (x_min + x_max) / 2.0 / win_w
        cy = (y_min + y_max) / 2.0 / win_h
        bw = (x_max - x_min + 1) / win_w
        bh = (y_max - y_min + 1) / win_h

        labels.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return labels


def main():
    # ------------------------------------------------------------------
    # Setup output folders
    # ------------------------------------------------------------------
    for split in ('train', 'val'):
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raster metadata
    # ------------------------------------------------------------------
    print(f"1. Reading raster metadata from: {RASTER_FILE}")
    with rasterio.open(RASTER_FILE) as src:
        transform  = src.transform
        raster_crs = src.crs
        width      = src.width
        height     = src.height
        band_count = src.count
    print(f"   {width}x{height} px | {band_count} bands | CRS: {raster_crs}")

    # ------------------------------------------------------------------
    # 2. Load and reproject points
    # ------------------------------------------------------------------
    print(f"\n2. Loading points from: {VECTOR_FILE}")
    gdf = gpd.read_file(VECTOR_FILE)
    if gdf.crs != raster_crs:
        print(f"   Reprojecting from {gdf.crs} → {raster_crs}")
        gdf = gdf.to_crs(raster_crs)
    print(f"   {len(gdf)} points loaded.")

    all_pixel_coords = get_pixel_coords(gdf, transform, width, height)
    valid_indices = [i for i, c in enumerate(all_pixel_coords) if c is not None]
    print(f"   {len(valid_indices)} points inside raster bounds.")

    # ------------------------------------------------------------------
    # 3. Build patch list and assign train/val split up front
    # ------------------------------------------------------------------
    random.seed(RANDOM_SEED)
    shuffled = valid_indices.copy()
    random.shuffle(shuffled)
    n_val    = max(1, int(len(shuffled) * VAL_SPLIT))
    val_set  = set(shuffled[:n_val])

    # ------------------------------------------------------------------
    # 4. Generate patches
    # ------------------------------------------------------------------
    print(f"\n3. Generating 1024×1024 patches...")
    half_w = WINDOW_SIZE // 2
    saved       = 0
    train_count = 0
    val_count   = 0
    skipped     = 0

    # Pre-build numpy array of all valid pixel coords for vectorised window lookups — O(N) per tile
    valid_px_array = np.array([all_pixel_coords[j] for j in valid_indices])  # (N, 2)

    with rasterio.open(RASTER_FILE, 'r') as src_img, \
         rasterio.open(MASK_FILE,   'r') as src_mask:

        for patch_idx, center_idx in enumerate(valid_indices):
            px, py = all_pixel_coords[center_idx]

            # --- Window bounds (border-safe) ---
            col_off = max(0, px - half_w)
            row_off = max(0, py - half_w)
            w = min(WINDOW_SIZE, width  - col_off)
            h = min(WINDOW_SIZE, height - row_off)
            window = Window(col_off=col_off, row_off=row_off, width=w, height=h)

            # --- Vectorised: find every point that falls inside this window ---
            x_locals = valid_px_array[:, 0] - col_off
            y_locals = valid_px_array[:, 1] - row_off
            inside   = (x_locals >= 0) & (x_locals < w) & (y_locals >= 0) & (y_locals < h)
            points_in_window = [
                (valid_indices[i], int(x_locals[i]), int(y_locals[i]))
                for i in np.where(inside)[0]
            ]

            # --- Read image patch ---
            if band_count >= 3:
                img_data = src_img.read([1, 2, 3], window=window)
            else:
                img_data = src_img.read(1, window=window)
                img_data = np.stack([img_data, img_data, img_data])

            img_patch = normalize_to_uint8(np.transpose(img_data, (1, 2, 0)))

            # --- Read mask patch ---
            mask_patch = src_mask.read(1, window=window)

            # --- Extract YOLO bounding boxes ---
            yolo_labels = get_yolo_boxes_in_window(
                mask_patch, points_in_window, w, h
            )

            if not yolo_labels:
                print(f"   [!] Patch {patch_idx + 1}: no valid boxes found. Skipping.")
                skipped += 1
                continue

            # --- Determine split ---
            split = 'val' if center_idx in val_set else 'train'
            patch_name = f"patch_{saved:05d}"

            # --- Save image ---
            img_path = os.path.join(OUTPUT_DIR, 'images', split, f"{patch_name}.png")
            Image.fromarray(img_patch).save(img_path)

            # --- Save YOLO label file ---
            lbl_path = os.path.join(OUTPUT_DIR, 'labels', split, f"{patch_name}.txt")
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

            if split == 'train':
                train_count += 1
            else:
                val_count += 1
            saved += 1
            print(f"   [{split}] patch_{saved - 1:05d} — center point {center_idx + 1} "
                  f"| window {w}×{h} | {len(yolo_labels)} box(es)")

    # ------------------------------------------------------------------
    # 5. Write dataset.yaml for YOLO11
    # ------------------------------------------------------------------
    yaml_path = os.path.join(OUTPUT_DIR, 'dataset.yaml')
    abs_output = os.path.abspath(OUTPUT_DIR)
    with open(yaml_path, 'w') as f:
        f.write(f"path: {abs_output}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['object']\n")

    print(f"\nDone!")
    print(f"  Patches saved : {saved}  (skipped: {skipped})")
    print(f"  Train / Val   : {train_count} / {val_count}")
    print(f"  Output dir    : {abs_output}/")
    print(f"  YOLO config   : {yaml_path}")


if __name__ == "__main__":
    main()
