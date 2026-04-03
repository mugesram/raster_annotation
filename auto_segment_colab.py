# ==========================================
# INSTALL DEPENDENCIES
# ==========================================
# !pip install ultralytics rasterio geopandas shapely -q
# !pip install git+https://github.com/facebookresearch/segment-anything-2.git -q
# !wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from contextlib import nullcontext
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==========================================
# CONFIGURATION
# ==========================================
RASTER_FILE     = 'input_raster.tif'
CONTAINER_FILE  = 'containers.geojson'     # polygon file — processing limited to inside these
OUTPUT_FILE     = 'auto_mask.tif'

YOLO_WEIGHTS    = 'runs/yolo11_raster_seg/weights/best.pt'
YOLO_CONF       = 0.5                      # detections below this are ignored

SAM2_CHECKPOINT = 'sam2.1_hiera_large.pt'
MODEL_CFG       = 'configs/sam2.1/sam2.1_hiera_l.yaml'

WINDOW_SIZE     = 1024
STRIDE          = 512   # 50% overlap so objects at tile edges are not missed


# ==========================================
# HELPERS
# ==========================================

def normalize_to_uint8(patch, global_min, global_max):
    if patch.dtype == np.uint8:
        return patch
    denom = float(global_max - global_min) if global_max > global_min else 1.0
    patch = (patch.astype(np.float32) - global_min) / denom * 255.0
    return np.clip(patch, 0, 255).astype(np.uint8)


def tile_geo_bbox(window_transform, w, h):
    """Return a Shapely box covering the geographic extent of a raster window."""
    minx = window_transform.c
    maxy = window_transform.f
    maxx = minx + window_transform.a * w
    miny = maxy + window_transform.e * h   # e is negative for north-up rasters
    return shapely_box(minx, miny, maxx, maxy)


# ==========================================
# MAIN
# ==========================================

def main():

    # ------------------------------------------------------------------
    # 1. Load raster metadata + global normalization stats
    # ------------------------------------------------------------------
    print(f"1. Reading raster: {RASTER_FILE}")
    with rasterio.open(RASTER_FILE) as src:
        profile    = src.profile
        transform  = src.transform
        raster_crs = src.crs
        width      = src.width
        height     = src.height
        band_count = src.count

        print("   Computing global normalization stats (downsampled)...")
        sample     = src.read(out_shape=(band_count, max(1, height // 20), max(1, width // 20)))
        global_min = float(sample.min())
        global_max = float(sample.max())

    print(f"   {width}×{height} px | {band_count} bands | CRS: {raster_crs}")

    # ------------------------------------------------------------------
    # 2. Load container polygons + compute pixel-space tile range
    # ------------------------------------------------------------------
    print(f"\n2. Loading containers: {CONTAINER_FILE}")
    gdf = gpd.read_file(CONTAINER_FILE)
    if gdf.crs != raster_crs:
        print(f"   Reprojecting from {gdf.crs} → {raster_crs}")
        gdf = gdf.to_crs(raster_crs)

    container_geoms  = [g for g in gdf.geometry if g is not None and g.is_valid]
    if not container_geoms:
        raise RuntimeError("No valid geometries found in container file.")
    container_union  = unary_union(container_geoms)
    print(f"   {len(container_geoms)} polygon(s) loaded.")

    # Convert container bounding box from geo → pixel space
    bounds           = container_union.bounds          # (minx, miny, maxx, maxy)
    col_min, row_max = ~transform * (bounds[0], bounds[1])
    col_max, row_min = ~transform * (bounds[2], bounds[3])
    col_min = max(0, int(col_min))
    row_min = max(0, int(row_min))
    col_max = min(width,  int(col_max))
    row_max = min(height, int(row_max))
    print(f"   Tile range — cols [{col_min}:{col_max}]  rows [{row_min}:{row_max}]")

    # ------------------------------------------------------------------
    # 3. Load YOLO + SAM2
    # ------------------------------------------------------------------
    print(f"\n3. Loading models...")
    yolo   = YOLO(YOLO_WEIGHTS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device : {device}")

    # Factory so each `with autocast():` gets a fresh context — safe across all PyTorch versions
    use_autocast = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()

    with autocast():
        sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
        predictor  = SAM2ImagePredictor(sam2_model)

    print("   YOLO11 + SAM2 Large ready.")

    # ------------------------------------------------------------------
    # 4. Prepare output raster (tiled GeoTIFF, zero RAM footprint)
    # ------------------------------------------------------------------
    print(f"\n4. Preparing output: {OUTPUT_FILE}")
    # Remove photometric tag — original RGB profile would conflict with single-band output
    profile.pop('photometric', None)
    profile.update(
        driver='GTiff',
        dtype=rasterio.uint8,
        count=1,
        compress='lzw',
        nodata=0,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(OUTPUT_FILE, 'w', **profile):
        pass   # initialise empty file

    # ------------------------------------------------------------------
    # 5. Systematic tiling — only inside containers
    # ------------------------------------------------------------------
    col_steps   = list(range(col_min, col_max, STRIDE))
    row_steps   = list(range(row_min, row_max, STRIDE))
    total_tiles = len(col_steps) * len(row_steps)

    print(f"\n5. Tiling: {total_tiles} candidate tiles "
          f"({len(col_steps)} cols × {len(row_steps)} rows) "
          f"| stride={STRIDE} | window={WINDOW_SIZE}")

    tile_idx  = 0
    processed = 0
    skipped   = 0

    with rasterio.open(RASTER_FILE, 'r') as src, \
         rasterio.open(OUTPUT_FILE, 'r+') as dst:

        for row_off in row_steps:
            for col_off in col_steps:
                tile_idx += 1

                # Clamp to raster edge
                w = min(WINDOW_SIZE, width  - col_off)
                h = min(WINDOW_SIZE, height - row_off)
                window           = Window(col_off=col_off, row_off=row_off, width=w, height=h)
                window_transform = src.window_transform(window)

                # ── Skip tiles with no container overlap ──────────────
                tile_geo = tile_geo_bbox(window_transform, w, h)
                if not container_union.intersects(tile_geo):
                    skipped += 1
                    continue

                # ── Read image patch ───────────────────────────────────
                if band_count >= 3:
                    img_data = src.read([1, 2, 3], window=window)
                else:
                    img_data = src.read(1, window=window)
                    img_data = np.stack([img_data, img_data, img_data])

                img_patch = normalize_to_uint8(
                    np.transpose(img_data, (1, 2, 0)), global_min, global_max
                )

                # ── YOLO detection ─────────────────────────────────────
                yolo_result = yolo.predict(
                    img_patch,
                    conf=YOLO_CONF,
                    verbose=False,
                    device=str(device),   # ultralytics expects string e.g. 'cuda' / 'cpu'
                )
                detections = yolo_result[0].boxes

                if detections is None or len(detections) == 0:
                    skipped += 1
                    continue

                boxes_xyxy = detections.xyxy.cpu().numpy()   # (N, 4) x1 y1 x2 y2
                confs      = detections.conf.cpu().numpy()   # (N,)

                # ── Container pixel mask for this window ───────────────
                # True  = inside a container polygon → keep mask pixels
                # False = outside containers         → zero out
                inside_container = ~geometry_mask(
                    container_geoms,
                    out_shape=(h, w),
                    transform=window_transform,
                    all_touched=False,
                )

                # ── SAM2 — one prediction per YOLO box ─────────────────
                combined_mask = dst.read(1, window=window)   # merge with existing

                with autocast():
                    predictor.set_image(img_patch)

                    for box, conf in zip(boxes_xyxy, confs):
                        masks, scores, _ = predictor.predict(
                            box=box[None],             # shape (1, 4)
                            multimask_output=False,
                        )
                        seg_mask = masks[0].astype(np.uint8) * 255

                        # Zero pixels outside containers before merging
                        seg_mask[~inside_container] = 0

                        combined_mask = np.maximum(combined_mask, seg_mask)

                dst.write(combined_mask, 1, window=window)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                processed += 1
                print(f"   [{tile_idx:>5}/{total_tiles}] "
                      f"col={col_off:<6} row={row_off:<6} | "
                      f"{len(boxes_xyxy)} box(es) | "
                      f"conf {confs.min():.2f}–{confs.max():.2f}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print(f"\nDone!")
    print(f"  Tiles processed : {processed}")
    print(f"  Tiles skipped   : {skipped}  (outside containers or no detections)")
    print(f"  Output          : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
