# modis_workflow


## Workflow Overview

This section documents the MODIS HDF4 ETL pipeline as implemented in `hdf_extract_clean_patchify_annotate_saveas_nc.py`. The workflow is as follows:

1. **Read Raw MODIS HDF4EOS Tiles**
   - Each tile is 2400x2400 pixels at 500m resolution.
   - 7 surface reflectance bands are extracted using `pyhdf`.

2. **Data Cleaning and Filtering**
   - Apply cloud masks, band quality masks, fill value removal, and scaling.
   - Cloud masks are extracted and resampled from 1200x1200 (1km) to match the 500m grid to align it with surfacereflectance.

3. **Mosaicking**
   - Cleaned tiles are mosaicked based on geolocation to form a 4800x4800 region (e.g., central Europe).

4. **Patchification and Labeling**
   - The mosaic is split into hxw spatial patches.
   - Each patches are labeled as 'cloudy'and 'clearsky'
   - Each patch is assigned a unique location ID and temporal stamp for downstream training and analysis.

5. **Saving Output**
   - Each patch contains 7 reflectance bands, a cloud mask, and 10 temporal snapshots.
   - Data is saved in NetCDF format with custom metadata for permanent storage.
