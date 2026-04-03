import os
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from contextlib import nullcontext
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==========================================
# CONFIGURATION
# ==========================================
# Set these to match the files you'll upload to Colab
RASTER_FILE = 'input_raster.tif'
VECTOR_FILE = 'points.geojson' # Or .shp, .gpkg
OUTPUT_FILE = 'sam2_segmentation_mask.tif'

SAM2_CHECKPOINT = "sam2_hiera_large.pt"
MODEL_CFG = "sam2_hiera_l.yaml"

# Defines the context window size that SAM2 will see around each point.
# 1024x1024 is an excellent balance for keeping VRAM requirements low
# while giving SAM2 enough context to see large objects.
WINDOW_SIZE = 1024  

def main():
    print(f"1. Loading metadata from large raster: {RASTER_FILE}")
    with rasterio.open(RASTER_FILE) as src:
        profile = src.profile
        transform = src.transform
        raster_crs = src.crs
        width = src.width
        height = src.height
        count = src.count
        
        # We calculate global min-max from a small downsampled read
        # so that different patches are normalized consistently.
        print("   Calculating global stats for image normalization...")
        try:
            # Try to grab a highly decimated quick look
            sample_data = src.read(out_shape=(count, max(1, height//20), max(1, width//20)))
        except Exception:
            # Fallback for weird formats
            sample_data = src.read(1, window=Window(0, 0, min(1000, width), min(1000, height)))
            
        global_min = sample_data.min()
        global_max = sample_data.max()

    print(f"--- Raster INFO: {width}x{height} pixels, {count} bands, CRS: {raster_crs}")

    print(f"\n2. Extracting geographic points from: {VECTOR_FILE}")
    gdf = gpd.read_file(VECTOR_FILE)
    if gdf.crs != raster_crs:
        print(f"   Reprojecting points from {gdf.crs} to {raster_crs} ...")
        gdf = gdf.to_crs(raster_crs)

    print("\n3. Initializing SAM2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    # Use bfloat16 autocast on capable GPUs (e.g. A100s in Colab), otherwise no-op
    use_autocast = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()

    with autocast_ctx:
        sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
        predictor = SAM2ImagePredictor(sam2_model)

    print(f"\n4. Preparing output raster {OUTPUT_FILE} ...")
    # We create the output file as a Tiled TIFF. 
    # This keeps RAM usage effectively zero when writing to a 1GB footprint!
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw',
        nodata=0,
        tiled=True,
        blockxsize=256,
        blockysize=256
    )
    
    with rasterio.open(OUTPUT_FILE, 'w', **profile) as dst:
        pass # Just initialize empty file

    print("\n5. Processing segmentation via memory-safe Windowed queries...")
    
    # Re-open the source and our new output mask file simultaneously
    with rasterio.open(RASTER_FILE, 'r') as src, rasterio.open(OUTPUT_FILE, 'r+') as dst:
        half_w = WINDOW_SIZE // 2
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
                
            # ~transform applies inverse mapping: Geo-coords -> Pixel-coords
            col, row_img = ~transform * (geom.x, geom.y)
            X_global, Y_global = int(col), int(row_img)
            
            if not (0 <= X_global < width and 0 <= Y_global < height):
                print(f"   [!] Point {idx+1} is outside raster bounds. Skipping.")
                continue

            # Calculate safe cropping window (handling borders securely)
            col_off = max(0, X_global - half_w)
            row_off = max(0, Y_global - half_w)
            w = min(WINDOW_SIZE, width - col_off)
            h = min(WINDOW_SIZE, height - row_off)
            
            window = Window(col_off=col_off, row_off=row_off, width=w, height=h)
            
            X_local = X_global - col_off
            Y_local = Y_global - row_off
            
            # LAAAZY LOAD JUST THIS SPECIFIC WINDOW
            if count >= 3:
                img_patch = src.read([1, 2, 3], window=window)
            else:
                img_patch = src.read(1, window=window)
                img_patch = np.stack([img_patch, img_patch, img_patch])
                
            img_patch = np.transpose(img_patch, (1, 2, 0))
            
            # Normalize to strictly uint8
            if img_patch.dtype != np.uint8:
                img_patch = img_patch.astype(np.float32)
                denom = (global_max - global_min) if global_max > global_min else 1.0
                img_patch = ((img_patch - global_min) / denom) * 255.0
                img_patch = np.clip(img_patch, 0, 255).astype(np.uint8)

            # Generate SAM2 Embeddings only for this 1024x1024 window
            with autocast_ctx:
                predictor.set_image(img_patch)

                pt_array = np.array([[X_local, Y_local]])
                lbl_array = np.array([1]) # 1 = Object to Segment

                masks, scores, logits = predictor.predict(
                    point_coords=pt_array,
                    point_labels=lbl_array,
                    multimask_output=False
                )

            best_mask = masks[0].astype(np.uint8) * 255

            # READ existing data in exactly this window (so close points can merge)
            existing_mask = dst.read(1, window=window)

            # COMBINE old and new mask seamlessly
            combined_mask = np.maximum(existing_mask, best_mask)

            # WRITE back to disk, never flooding RAM!
            dst.write(combined_mask, 1, window=window)

            # Release fragmented VRAM between iterations to prevent OOM on large point sets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"   Processed point {idx+1}/{len(gdf)} - Crop: {w}x{h} - Accuracy Score: {scores[0]:.2f}")

    print("\nDone! The large mask is saved as an efficient Tiled GeoTIFF.")

if __name__ == "__main__":
    main()
