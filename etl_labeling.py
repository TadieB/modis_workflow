import os
import sys
import time
import re
import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import Resampling
from pyhdf.SD import SD, SDC
import logging
from datetime import datetime
import gc

# Set up logging
def setup_logging(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"modis_processing_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file

# ------------------ Configuration ------------------
CONFIG = {
    "fill_value": [-28672, 787410671],  # [reflectance int16, band quality uint32], 
    # "cloudmask_fillvalue": 65535,         # no for MOD09GA
    # "cloudmask_valid_range": (0, 57343),  # no for MOD09GA
    "valid_range": (-100, 16000),
    "scale_factor": 0.0001,
    "patch_size": 32,
    "required_bands": [1, 2, 3, 4, 5, 6, 7],
    "cloud_threshold": 0.2 # originally cloudstate data = uint32; binary mask = uint8 /0 or 1/ processed
}

# ------------------ Helper Functions ------------------

def extract_tile_id(filename: str) -> str:
    tokens = re.split(r'[_.]', filename)
    for token in tokens:
        token = token.lower()
        if re.fullmatch(r'h\d{2}v\d{2}', token):
            return token
    raise RuntimeError(f"Tile id not found in filename: {filename}")

def resample_cloud_mask(cloud_mask, scale_factor=2):
    # Upsample the cloud mask by replicating each pixel in both dimensions.
    repeated_rows = np.repeat(cloud_mask, scale_factor, axis=0)
    resampled = np.repeat(repeated_rows, scale_factor, axis=1)
    return resampled

def read_modis_data(hdf_path: str) -> dict:
    try:
        hdf = SD(hdf_path, SDC.READ)
        data_fields = {}
        # Read surface reflectance bands.
        for band in CONFIG["required_bands"]:
            band_name = f"sur_refl_b0{band}_1"
            data_fields[f"band{band}"] = hdf.select(band_name).get().astype(np.int16)
        # Read quality data.
        data_fields["quality"] = hdf.select("QC_500m_1").get().astype(np.uint32)

        # Read and process cloud mask.
        csq = hdf.select("state_1km_1").get().astype(np.uint32)  # 1073741824, 32 bit unsigned int or positive, but userguide states 16bit unsigned, fillvalue 65535 which should be wrong.
        cloud_state = csq & 0b11
        cloudy_mask = (cloud_state == 1)
        cloud_mask = cloudy_mask.astype(np.uint8)
        # Upsample cloud mask from 1km to 500m resolution.
        data_fields["cloud_mask"] = resample_cloud_mask(cloud_mask, scale_factor=2)
        hdf.end()
        return {"data": data_fields, "source": hdf_path}
    except Exception as e:
        raise RuntimeError(f"Error reading HDF file {hdf_path}: {str(e)}")

def process_tile(modis_data: dict) -> xr.DataArray:
    data = modis_data["data"]
    band_qa = data["quality"]
    qa_fill_mask = (band_qa == CONFIG["fill_value"][1])
    # Create quality control masks for each band.
    qc_masks = {
        f"band{band}": (((band_qa >> (2 + 4*(band-1))) & 0b1111) == 0) & ~qa_fill_mask
        for band in CONFIG["required_bands"]
    }
    # Reject tile if no valid pixels are present in band 1/we can check all bands/.
    if not np.any(qc_masks["band1"]):
        return None
    processed = []
    for band in CONFIG["required_bands"]:
        arr = data[f"band{band}"]
        clean = np.where(arr == CONFIG["fill_value"][0], np.nan, arr)
        clean = np.where((clean >= CONFIG["valid_range"][0]) & (clean <= CONFIG["valid_range"][1]), clean, np.nan)
        masked = np.where(qc_masks[f"band{band}"], clean * CONFIG["scale_factor"], np.nan) # scale highest quality pixels of each band, the rest = nan.
        processed.append(masked)
    # Stack processed bands and append cloud mask as an extra band.
    np_stack = np.stack(processed, axis=-1)
    combined = np.concatenate([np_stack, data["cloud_mask"][..., np.newaxis]], axis=-1)
    return xr.DataArray(
        combined,
        dims=('y', 'x', 'band'),
        coords={'band': [f'band{b}' for b in CONFIG["required_bands"]] + ['cloud_mask']},
        attrs={'source_file': os.path.basename(modis_data['source'])}
    )

def process_file(input_path: str) -> xr.DataArray:
    modis_data = read_modis_data(input_path)
    processed_da = process_tile(modis_data)
    if processed_da is None:
        logging.warning(f"Rejected tile: {input_path}")
    return processed_da

def mosaic_tiles(day_dir: str) -> xr.DataArray:
    """
    Mosaic four tiles from a day directory.
    Required tile IDs: "h18v03", "h18v04", "h19v03", "h19v04"
    """
    files = [os.path.join(day_dir, f) for f in os.listdir(day_dir) if f.lower().endswith('.hdf')]
    tile_dict = {}
    for fp in files:
        try:
            tid = extract_tile_id(fp)
            logging.info(f"Found file {fp} with tile ID: {tid}")
            tile = process_file(fp)
            if tile is not None:
                tile_dict[tid] = tile
        except Exception as e:
            logging.error(f"Error processing file {fp}: {str(e)}")
    required_tiles = ["h18v03", "h18v04", "h19v03", "h19v04"]
    for tid in required_tiles:
        if tid not in tile_dict:
            logging.warning(f"Skipping day {day_dir}: Missing required tile {tid}")
            return None
    computed_tiles = [tile_dict[tid] for tid in required_tiles]
    if any(tile is None for tile in computed_tiles):
        logging.warning(f"Skipping day {day_dir}: One or more tiles computed as None.")
        return None
    # Mosaic by concatenating left column tiles vertically and right column tiles vertically, then merging horizontally.
    left_col = xr.concat([computed_tiles[0], computed_tiles[1]], dim='y')
    right_col = xr.concat([computed_tiles[2], computed_tiles[3]], dim='y')
    mosaic = xr.concat([left_col, right_col], dim='x')
    return mosaic

def save_mosaics(mosaics: list, output_dir: str, base_name: str):
    os.makedirs(output_dir, exist_ok=True)
    for mosaic in mosaics:
        day = str(mosaic.coords['time'].values[0])
        ds = mosaic.to_dataset(name='surface_reflectance')
        ds['band'] = ds.band.astype(str)
        output_path = os.path.join(output_dir, f"{base_name}_mosaic_{day}.nc")
        encoding = {'surface_reflectance': {'zlib': True, 'dtype': 'float32', '_FillValue': np.nan}}
        ds.to_netcdf(output_path, encoding=encoding)
        logging.info(f"Saved mosaic for day {day} at {output_path}")

def generate_patches(da: xr.DataArray) -> list:
    """
    Generate spatial patches from a 4D DataArray.
    The input DataArray must have dimensions (time, y, x, band), where the
    'time' dimension includes all days. Patches are extracted along the spatial
    dimensions (y, x) so that each patch retains the entire temporal sequence.
    (Each patch will have shape (time, patch_size, patch_size, band).)
    """
    patches = []
    patch_size = CONFIG["patch_size"]
    y_size, x_size = da.sizes['y'], da.sizes['x']
    patch_num = 0
    for y in range(0, y_size, patch_size):
        for x in range(0, x_size, patch_size):
            patch = da.isel(y=slice(y, y + patch_size),
                            x=slice(x, x + patch_size))
            if patch.sizes['y'] == patch_size and patch.sizes['x'] == patch_size:
                # Store time as a tuple (hashable) rather than a list.
                patch = patch.assign_attrs({
                    'patch_num': patch_num,
                    'origin': (y, x),
                    'time': tuple(patch.coords['time'].values.tolist())
                })
                patches.append((patch_num, patch))
                patch_num += 1
    return patches

def save_patches(patches: list, output_dir: str, base_name: str):
    os.makedirs(output_dir, exist_ok=True)
    for num, patch in patches:
        # Get location info from patch attributes
        origin = patch.attrs.get('origin')
        y, x = origin  # Extract coordinates
        
        # Create more descriptive filename including location
        filename = f"{base_name}_y{y}_x{x}_patch_{num:04d}.nc"
        output_path = os.path.join(output_dir, filename)
        
        # Save with dimensions (time, 32, 32, 8)
        ds = patch.to_dataset(name='surface_reflectance')
        ds['band'] = ds.band.astype(str)
        encoding = {'surface_reflectance': {'zlib': True, 'dtype': 'float32', '_FillValue': np.nan}}
        ds.to_netcdf(output_path, encoding=encoding)
        
        logging.info(f"Saved patch file: {filename}")
        logging.info(f"  Location: y={y}, x={x}")
        logging.info(f"  Shape: {patch.shape}")  # Should show (time, 32, 32, 8)

def verify_location_consistency(group):
    locations = set(patch.attrs.get('origin') for patch in group)
    if len(locations) > 1:
        logging.error(f"Location inconsistency detected: {locations}")
        return False
    return True

def generate_supervised_pairs(patches, threshold=CONFIG["cloud_threshold"]):
    if len(patches) == 0:  # Need at least one patch
        logging.error("No patches provided")
        return []
    
    groups = {}
    for patch in patches:
        key = patch.attrs.get('origin')
        groups.setdefault(key, []).append(patch)
    
    logging.info(f"Grouped {len(patches)} patches into {len(groups)} spatial groups")
    logging.info(f"Cloud threshold for clean patches: {threshold*100}%")
    
    samples = []
    location_stats = {}
    
    for key, group in groups.items():
        y, x = key  # Location coordinates
        if not verify_location_consistency(group):
            logging.error(f"Skipping inconsistent location y={y}, x={x}")
            continue
        
        # Check if the patch contains multiple days in its time dimension
        sample_patch = group[0]  # We should only have one patch per location
        
        # Extract all days from the patch's time coordinate
        time_values = sample_patch.coords['time'].values
        days = [str(t) for t in time_values]
        expected_days = len(time_values)  # Use actual time dimension instead of hardcoded value
        
        # Debug: Check how many days we have in this patch's time dimension
        logging.info(f"Location ({y},{x}) has {len(days)} days in time dimension")
        logging.info(f"Days at this location: {days}")
        
        # Skip if we don't have the expected number of days for this location
        if len(days) < expected_days:
            logging.warning(f"Location ({y},{x}) has only {len(days)} days in time dimension, need {expected_days}. Skipping.")
            continue
            
        logging.info(f"\n=== Processing Location y={y}, x={x} ===")
        
        # Process each day in the time dimension
        all_patches = []
        for day_idx, day in enumerate(days):
            # Extract cloud mask for this day
            cloud_mask = sample_patch.isel(time=day_idx, band=-1).values
            total_pixels = np.prod(cloud_mask.shape)
            cloud_pct = np.sum(cloud_mask) / total_pixels
            if not np.isfinite(cloud_pct):
                logging.warning(f"Invalid cloud percentage at location y={y}, x={x}, day {day}")
                continue
            
            # Store the patch, cloud percentage, and day
            all_patches.append((sample_patch, cloud_pct, day))
            logging.info(f"  Day {day}: cloud cover = {cloud_pct*100:.1f}%")
        
        # Also use the expected_days variable here
        if len(all_patches) < expected_days:
            logging.warning(f"Location ({y},{x}) has only {len(all_patches)} valid days, need {expected_days}. Skipping.")
            continue
            
        clean_patches = [(p, cp, t) for p, cp, t in all_patches if cp <= threshold]
        
        if clean_patches:
            target_patch, target_cloud_pct, target_day = min(clean_patches, key=lambda x: x[1])
            target_day_idx = days.index(target_day)  # Get index of target day
            
            logging.info(f"\n=== Location Statistics (y={y}, x={x}) ===")
            logging.info(f"  Total available patches: {len(all_patches)}")
            logging.info(f"  Clean patches (≤{threshold*100}%): {len(clean_patches)}")
            logging.info(f"  Selected target:")
            logging.info(f"    - Day: {target_day}")
            logging.info(f"    - Cloud cover: {target_cloud_pct*100:.1f}%")
            logging.info(f"  Input-Target Pairs:")
            logging.info(f"    - Target from day {target_day} will be paired with {len(all_patches)} inputs")
            logging.info(f"    - Input days and their cloud cover:")
            
            for day, cloud_pct in zip([t for _, _, t in all_patches], 
                                      [cp for _, cp, _ in all_patches]):
                logging.info(f"      Input day {day}: {cloud_pct*100:.1f}% clouds")
            
            # Store detailed statistics for this location
            location_stats[key] = {
                'coordinates': (y, x),
                'total_patches': len(all_patches),
                'clean_patches': len(clean_patches),
                'target_day': target_day,
                'target_cloud_pct': target_cloud_pct,
                'input_days': [t for _, _, t in all_patches],
                'input_cloud_pcts': [cp for _, cp, _ in all_patches],
                'num_input_target_pairs': len(all_patches)
            }
            
            # Create sample with all days
            # No need to create a new input_data - just use the original patch
            input_data = sample_patch
            
            # Convert days to integers for the sample
            all_days = [int(d) for d in days]
            
            logging.info(f"Input data shape: {input_data.shape}")
            logging.info(f"Number of days collected: {len(all_days)}")
            
            samples.append({
                'input': input_data,  # Already has shape (10, 32, 32, 7+1)
                'target': input_data.isel(time=target_day_idx, band=slice(0, len(CONFIG['required_bands']))).expand_dims('time'),  # Get target day
                'mask': input_data.isel(band=-1),  # Cloud mask for all days
                'cloud_pcts': [cp for _, cp, _ in all_patches],
                'target_cloud_pct': target_cloud_pct,
                'days': all_days,
                'target_day': int(target_day)
            })

            # Add debug logging
            logging.info(f"Sample shapes and time information:")
            logging.info(f"  Input: {input_data.shape}, days={all_days}")
            logging.info(f"  Target: {input_data.isel(time=target_day_idx, band=slice(0, len(CONFIG['required_bands']))).expand_dims('time').shape}, day={target_day}")
            logging.info(f"  Mask: {input_data.isel(band=-1).shape}")
        else:
            logging.info(f"\n=== Location Statistics (y={y}, x={x}) ===")
            logging.info(f"  No valid target found - all patches above {threshold*100}% cloud cover")
            logging.info(f"  Available patches: {len(all_patches)}")
            logging.info("  Cloud cover per day:")
            for (_, cp, t) in all_patches:
                logging.info(f"    Day {t}: {cp*100:.1f}%")

    # Final summary statistics
    logging.info("\n=== Overall Location Statistics Summary ===")
    total_pairs = 0
    for coords, stats in location_stats.items():
        y, x = coords
        total_pairs += stats['num_input_target_pairs']
        logging.info(f"\nLocation y={y}, x={x}:")
        logging.info(f"  - Target from day {stats['target_day']} ({stats['target_cloud_pct']*100:.1f}% clouds)")
        logging.info(f"  - Paired with {stats['num_input_target_pairs']} inputs")
        logging.info(f"  - Input days: {', '.join(map(str, stats['input_days']))}")

    logging.info(f"\nTotal locations with valid pairs: {len(location_stats)}")
    logging.info(f"Total input-target pairs across all locations: {total_pairs}")
    
    # Let's add a final validation
    for i, sample in enumerate(samples):
        logging.info(f"Final sample {i+1} validation:")
        logging.info(f"  Input shape: {sample['input'].shape} (should be {expected_days}, 32, 32, 8)")
        logging.info(f"  Target shape: {sample['target'].shape} (should be 1, 32, 32, 7)")
        logging.info(f"  Mask shape: {sample['mask'].shape} (should be {expected_days}, 32, 32)")
        logging.info(f"  Days: {len(sample['days'])} days (should be {expected_days})")

    return samples

def save_supervised_sample(sample, base_output_dir):
    # Get the expected number of days from the input data shape
    expected_days = sample['input'].shape[0]
    
    # First check if we have all expected days collected
    if len(sample['days']) < expected_days:
        logging.info(f"Waiting for more days... Currently have {len(sample['days'])} days, need {expected_days}")
        return False

    # Now validate dimensions since we have all days
    input_shape = sample['input'].shape
    input_time = sample['days']
    if input_shape[0] != len(input_time):
        logging.error(f"Input time mismatch: data shape {input_shape[0]} != coordinate length {len(input_time)}")
        return False

    # 1. TARGET validation
    target_shape = sample['target'].shape  # Should be (1, 32, 32, 7)
    target_time = [sample['target_day']]   # Single day
    if target_shape[0] != len(target_time):
        logging.error(f"Target time mismatch: data shape {target_shape[0]} != coordinate length {len(target_time)}")
        return False

    # 2. INPUT validation
    input_shape = sample['input'].shape    # Should be (10, 32, 32, 7)
    input_time = sample['days']           # List of 10 days
    if input_shape[0] != len(input_time):
        logging.error(f"Input time mismatch: data shape {input_shape[0]} != coordinate length {len(input_time)}")
        return False

    # 3. MASK validation
    mask_shape = sample['mask'].shape     # Should be (10, 32, 32)
    mask_time = sample['days']            # Same 10 days as input
    if mask_shape[0] != len(mask_time):
        logging.error(f"Mask time mismatch: data shape {mask_shape[0]} != coordinate length {len(mask_time)}")
        return False

    # Create datasets with explicit dimension matching
    ds_target = xr.Dataset({
        'target': xr.DataArray(
            sample['target'].values,
            dims=['time', 'y', 'x', 'band'],
            coords={
                'time': [int(sample['target_day'])],  # Single day number (e.g., [165])
                'y': range(32),
                'x': range(32),
                'band': range(7)
            }
        )
    })

    ds_input = xr.Dataset({
        'input': xr.DataArray(
            # Use only the first 7 bands (reflectance) for input, exclude cloud mask
            sample['input'].isel(band=slice(0, len(CONFIG['required_bands']))).values,
            dims=['time', 'y', 'x', 'band'],
            coords={
                'time': input_time,
                'y': range(32),
                'x': range(32),
                'band': range(7)
            }
        )
    })

    ds_mask = xr.Dataset({
        'cloud_mask': xr.DataArray(
            sample['mask'].values,
            dims=['time', 'y', 'x'],
            coords={
                'time': [int(d) for d in sample['days']],  # Same sequential days as input
                'y': range(32),
                'x': range(32)
            }
        )
    })

    # Log the final validation
    logging.info("Final dimension check:")
    logging.info(f"  Target - data: {ds_target.target.shape}, time coord: {len(ds_target.time)}")
    logging.info(f"  Input  - data: {ds_input.input.shape}, time coord: {len(ds_input.time)}")
    logging.info(f"  Mask   - data: {ds_mask.cloud_mask.shape}, time coord: {len(ds_mask.time)}")

    logging.info(f"Processing sample for saving")
    logging.info(f"  - Input has shape {input_shape} with {expected_days} time points")
    logging.info(f"  - Target has shape {target_shape} with 1 time point (day {sample['target_day']})")
    logging.info(f"  - Mask has shape {mask_shape} with {expected_days} time points")
    logging.info(f"All dimensions validated, proceeding to save files")

    try:
        # Get location info
        origin = sample['target'].attrs.get('origin')
        patch_num = sample['target'].attrs.get('patch_num')
        location_id = f"tile_{origin[0]}_{origin[1]}_patch_{patch_num:04d}"

        # Create directories
        target_dir = os.path.join(base_output_dir, "target")
        inputs_dir = os.path.join(base_output_dir, "inputs")
        cloudmask_dir = os.path.join(base_output_dir, "cloudmask")
        
        for d in [target_dir, inputs_dir, cloudmask_dir]:
            os.makedirs(d, exist_ok=True)

        # TARGET: Single day, single cloud cover
        target_cloud = f"{round(sample['target_cloud_pct'] * 100, 1)}"
        target_fname = f"{location_id}_target_cloud{target_cloud}_day{sample['target_day']}.nc"
        
        # INPUT/MASK: Multiple days
        input_clouds = [f"{round(cp * 100, 1)}" for cp in sample['cloud_pcts']]
        input_days = sample['days']
        
        if len(input_days) > 4:
            input_clouds_str = f"{input_clouds[0]}-{input_clouds[-1]}"
            input_days_str = f"{input_days[0]}_{input_days[-1]}"
        else:
            input_clouds_str = '-'.join(input_clouds)
            input_days_str = '_'.join(map(str, input_days))
            
        input_fname = f"{location_id}_input_clouds{input_clouds_str}_days{input_days_str}.nc"
        mask_fname = f"{location_id}_mask_clouds{input_clouds_str}_days{input_days_str}.nc"

        # Save files with correct filenames
        ds_target.to_netcdf(os.path.join(target_dir, target_fname))
        ds_input.to_netcdf(os.path.join(inputs_dir, input_fname))
        ds_mask.to_netcdf(os.path.join(cloudmask_dir, mask_fname))

        # Verification logging
        logging.info(f"Successfully saved sample {location_id}:")
        logging.info(f"  Target: shape={ds_target.target.shape}, time={ds_target.time.values}")
        logging.info(f"  Input: shape={ds_input.input.shape}, time={ds_input.time.values}")
        logging.info(f"  Mask: shape={ds_mask.cloud_mask.shape}, time={ds_mask.time.values}")

        return True

    except Exception as e:
        logging.error(f"Error saving sample {location_id}: {str(e)}")
        logging.error(f"Time coordinates when error occurred:")
        logging.error(f"  Input days: {sample['days']}")
        logging.error(f"  Target day: {sample['target_day']}")
        return False

def save_supervised_samples(samples, base_output_dir):
    failed_samples = []
    successful = 0  # Add counter for successful saves
    
    for sample in samples:
        try:
            success = save_supervised_sample(sample, base_output_dir)
            if success:
                successful += 1  # Increment on successful save
            else:
                failed_samples.append(sample)
        except Exception as e:
            logging.error(f"Failed to save sample: {e}")
            failed_samples.append(sample)
    
    if failed_samples:
        logging.warning(f"Failed to save {len(failed_samples)} samples")
    
    return successful  # Return the count of successful saves

def process_directory(root_dir: str, patches_output_dir: str, mosaics_output_dir: str, supervised_output_dir: str):
    # Add more detailed logging
    logging.info(f"Processing directory: {root_dir}")
    logging.info(f"Cloud threshold: {CONFIG['cloud_threshold']*100}%")
    logging.info(f"Patch size: {CONFIG['patch_size']}x{CONFIG['patch_size']}")
    
    # Add validation for output directories
    for d in [patches_output_dir, mosaics_output_dir, supervised_output_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            logging.info(f"Created output directory: {d}")
        
    base_name = os.path.basename(os.path.normpath(root_dir))
    
    # Fix: Remove the 5-day limitation for production use
    day_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])[10:]
    
    logging.info(f"Found {len(day_dirs)} day directories to process")
    
    my_day_mosaics = []
    successful_days = 0
    for day in day_dirs:
        day_dir = os.path.join(root_dir, day)
        try:
            mosaic = mosaic_tiles(day_dir)
            if mosaic is not None:
                # Expand mosaic to include a 'time' coordinate (per day).
                mosaic = mosaic.expand_dims(time=[day])
                my_day_mosaics.append(mosaic)
                successful_days += 1
                logging.info(f"Processed day {day} ({successful_days}/{len(day_dirs)})")
            else:
                logging.warning(f"No valid mosaic for {day_dir}")
        except Exception as e:
            logging.error(f"Error processing {day_dir}: {str(e)}")
    
    # Fix: Add check for minimum valid data
    if not my_day_mosaics:
        logging.error("No valid day mosaics found. Cannot proceed.")
        return
    
    logging.info(f"Successfully processed {successful_days}/{len(day_dirs)} days")
    
    # Sort mosaics by time
    my_day_mosaics = sorted(my_day_mosaics, key=lambda da: da.coords['time'].values[0])
    save_mosaics(my_day_mosaics, mosaics_output_dir, base_name)
    
    # Concatenate daily mosaics along the time dimension and rearrange dimensions.
    combined_da = xr.concat(my_day_mosaics, dim="time").transpose("time", "y", "x", "band")
    
    # Generate patches along the spatial dimensions only.
    patches = generate_patches(combined_da)
    save_patches(patches, patches_output_dir, base_name)
    logging.info(f"Aggregated mosaic and patches saved.")
    logging.info(f"Generated {len(patches)} total patches")
    
    # Fix: Check if we have enough patches
    if len(patches) == 0:
        logging.error("No patches were generated. Cannot create supervised pairs.")
        return
    
    patch_list = [p for _, p in patches]
    logging.info(f"Processing {len(patch_list)} patches for supervised pairs")
    
    supervised_samples = generate_supervised_pairs(patch_list, threshold=CONFIG["cloud_threshold"])
    
    # Fix: Add check for minimum supervised samples
    if len(supervised_samples) == 0:
        logging.error("No supervised pairs were generated. Check cloud threshold and data quality.")
        return
    
    logging.info(f"Generated {len(supervised_samples)} supervised samples")
    
    # Make sure to capture the return value from save_supervised_samples
    success_count = save_supervised_samples(supervised_samples, supervised_output_dir)
    
    # Fix: Add final validation step
    target_dir = os.path.join(supervised_output_dir, "target")
    inputs_dir = os.path.join(supervised_output_dir, "inputs")
    cloudmask_dir = os.path.join(supervised_output_dir, "cloudmask")
    
    # Check if all directories exist and contain files
    target_files = os.listdir(target_dir) if os.path.exists(target_dir) else []
    inputs_files = os.listdir(inputs_dir) if os.path.exists(inputs_dir) else []
    mask_files = os.listdir(cloudmask_dir) if os.path.exists(cloudmask_dir) else []
    
    if len(target_files) > 0 and len(inputs_files) > 0 and len(mask_files) > 0:
        logging.info(f"Successfully created labeled dataset with {len(target_files)} samples")
        logging.info(f"- Target files: {len(target_files)}")
        logging.info(f"- Input files: {len(inputs_files)}")
        logging.info(f"- Mask files: {len(mask_files)}")
    else:
        logging.error(f"Failed to create complete labeled dataset: target={len(target_files)}, inputs={len(inputs_files)}, masks={len(mask_files)}")
        
    # Update summary log to use success_count
    logging.info(f"Summary: Processed {successful_days} days, generated {len(patches)} patches, created {success_count} supervised pairs")

    # After creating my_day_mosaics in process_directory:
    logging.info(f"Days processed: {[m.coords['time'].values[0] for m in my_day_mosaics]}")
    logging.info(f"Expected 10 days, got {len(my_day_mosaics)}")

    # After collecting all day mosaics in process_directory:
    if len(my_day_mosaics) < 10:
        logging.error(f"Not enough valid days: Expected 10, got {len(my_day_mosaics)}. Need all days for proper time dimension.")
        # Optionally, we could handle this by adjusting expectations:
        # CONFIG["expected_days"] = len(my_day_mosaics)
        # logging.warning(f"Adjusting expected days to {CONFIG['expected_days']}")
        
        # For now, just return and abort if we don't have all 10 days
        if len(my_day_mosaics) < 2:
            logging.error("Insufficient days to create supervised pairs. Aborting.")
            return

def create_filename(location_id, data_type, clouds, timestamps):
    """Create filename with smart truncation for long day sequences.
    
    Args:
        location_id: e.g., 'tile_2304_4032_patch_0123'
        data_type: one of 'input', 'target', or 'mask'
        clouds: list of cloud cover percentages
        timestamps: list of day numbers
    """
    # Validate data_type
    if data_type not in ['input', 'target', 'mask']:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'input', 'target', or 'mask'")
    
    # First try full filename with data type label
    fname = f"{location_id}_{data_type}_clouds{'-'.join(clouds)}_{timestamps}.nc"
    
    if len(fname) <= 255:  # If filename is acceptable length, use it as is
        return fname
        
    # Need to truncate - first compress the timestamps
    timestamps = [str(t) for t in timestamps]
    if len(timestamps) > 3:  # If we have many timestamps
        # Convert sequence like ['165', '166', '167', '168'] to '165_168'
        compressed_time = f"{timestamps[0]}_{timestamps[-1]}"
    else:
        compressed_time = '_'.join(timestamps)
    
    # Try filename with compressed timestamps
    fname = f"{location_id}_{data_type}_clouds{'-'.join(clouds)}_{compressed_time}.nc"
    
    if len(fname) <= 255:
        return fname
        
    # If still too long, round cloud percentages to fewer decimal places
    clouds = [f"{float(c):.1f}" for c in clouds]  # Reduce to 1 decimal place
    fname = f"{location_id}_{data_type}_clouds{'-'.join(clouds)}_{compressed_time}.nc"
    
    if len(fname) <= 255:
        return fname
        
    # If still too long, keep only first and last cloud percentage
    if len(clouds) > 2:
        clouds_str = f"{clouds[0]}-{clouds[-1]}"
    else:
        clouds_str = '-'.join(clouds)
    
    fname = f"{location_id}_{data_type}_clouds{clouds_str}_{compressed_time}.nc"
    
    logging.info(f"Truncated long filename to: {fname}")
    return fname


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='MODIS HDF Supervised Dataset Generator: Mosaic, Patch, and Supervised Pair Processing'
    )
    parser.add_argument('--input', default="./your_hdf_dir",
                        help='Input root directory containing day directories')
    parser.add_argument('--patch_output', default="./your_dir_for_patches", #unlabeled
                        help='Output directory for patches')
    parser.add_argument('--mosaic_output', default="./your_dir_for_mosaics",help='Output directory for mosaics')
    parser.add_argument('--supervised_output', default="./your_dir_for_labeled_data", help='Output directory for supervised pairs')
    parser.add_argument('--log_dir', default="./logs", help='Directory for log files')
    args = parser.parse_args()

    # Set up logging
    log_file = setup_logging(args.log_dir)
    logging.info(f"Starting MODIS processing with input directory: {args.input}")
    
    start_time = time.time()
    process_directory(args.input, args.patch_output, args.mosaic_output, args.supervised_output)
    elapsed_time = time.time() - start_time
    logging.info(f"Processing complete in {elapsed_time:.2f} seconds.")
    logging.info(f"Strict threshold <= 20%, Log file saved to: {log_file}")
