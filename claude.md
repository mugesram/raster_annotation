# Project Summary: SAM 2 Segmentation for Large Rasters

## Requirements
1. **Inputs:** 
   - A large georeferenced raster file (~1GB or more).
   - A point vector file (e.g., GeoJSON, GeoPackage, Shapefile) generated in QGIS by clicking over objects of interest.
2. **Core Task:** Run Meta's Segment Anything Model 2 (SAM 2) using those geographic points as prompts to generate segmentation masks.
3. **Execution Environment:** Google Colab (GPU Runtime). 
4. **Constraints:** 
   - The code must be strictly contained in a `.py` script format.
   - The memory architecture must handle loading 1GB+ imagery without triggering Out-Of-Memory (OOM) crashes on Colab.
   - The final output raster must be accurately georeferenced, preserving the spatial integrity of the input file so it opens precisely in place within QGIS.

## What Has Been Done
1. **Python Script Generation (`sam2_qgis_colab.py`):**
   - Wrote a self-contained, heavily commented Python pipeline explicitly engineered for Colab integration.
   - Intercepts QGIS vector coordinates and applies inverse Affine matrices to perfectly translate real-world clicks into X, Y pixel grids. 

2. **Aggressive Memory Optimization (Windowed Queries):**
   - Rather than attempting to cast 1GB to a float32 array in memory, implemented a memory-safe `rasterio.windows` approach.
   - The script iterates through the input vector points, calculating a localized 1024x1024 pixel bounding box around each point.
   - SAM 2 receives only these extracted small RGB crops, minimizing GPU VRAM footprint and entirely bypassing CPU RAM bottlenecks.

3. **Progressive Tiled Georeferencing:**
   - Extracted spatial metadata (the Coordinate Reference System and Transform matrix) from the original input raster.
   - Programmed the script to generate an empty "Tiled" GeoTIFF mask with identical spatial metadata to guarantee perfect QGIS realignment.
   - Seamlessly writes the localized SAM 2 mask predictions continuously back to the hard drive on-the-fly, ensuring the massive mask file never fills up active memory.

4. **Architectural Evaluation:**
   - Discussed the potential of utilizing third-party wrappers like `samgeo` (`segment-geospatial`). Concluded that although excellent, the direct memory abstraction afforded by custom windowing was a safer requirement match for specifically massive 1GB inputs.

5. **Bug Fixes & Robustness Hardening:**
   - Fixed a broken `torch.autocast` usage where `.__enter__()` was called manually without a matching `.__exit__()`, leaking the context. Replaced with a proper `with` context manager using `contextlib.nullcontext()` as a no-op fallback on CPU or older GPUs.
   - Added `torch.cuda.empty_cache()` after each point's write step to release fragmented VRAM between iterations, preventing OOM crashes on large point sets.
   - Both model initialization (`build_sam2`) and all per-point inference calls (`set_image`, `predict`) are now correctly scoped inside the `autocast_ctx` context manager.
