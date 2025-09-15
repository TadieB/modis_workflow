
# Multispectral / Multitemporal Cloud-Gap Imputation — Data Pipeline

This repository builds a NetCDF dataset for cloud-gap imputation from multispectral and multitemporal satellite products.

---

## Install

```bash
# (optional) create venv
python3 -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

Dependencies include: `numpy`, `xarray`, `pyhdf`, `netCDF4`, `rasterio`, `globus-compute-sdk`, `globus-compute-endpoint`.

---

## Workflow

1. **Download data**

   ```bash
   python3 multispectral_multitemporal_downloader.py
   ```
2. **Run ETL + labeling**

   ```bash
   python3 etl_labeling.py
   ```
3. **Result**: NetCDF dataset with multispectral, multitemporal stacks labeled as *cloudy* or *clear-sky*.

---

## Configuration

Adjust the config section in the scripts to change:

* products (e.g., MOD09, MOD35\_L2, MOD03)
* time window (years/dates)
* spatial sizes
* multispectral bands
* region/tiles
* metrics or labeling options
* directories

---

