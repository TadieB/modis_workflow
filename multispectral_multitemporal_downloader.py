# 30days x 10 files x 300MB avg = 90 GB files to be downloaded!!!!!!!

import os
import pathlib
import logging
import argparse
from typing import List
import yaml
import sys

# Parsl imports
import parsl
from parsl.config import Config
from parsl.providers import SlurmProvider, LocalProvider
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_interface
from parsl.executors import HighThroughputExecutor 
from parsl.channels import LocalChannel

# Globus SDK imports
import globus_sdk
import globus_sdk.scopes
from globus_compute_sdk import Executor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger("MOD09GA_downloader")

def get_args(verbose=True):
    p = argparse.ArgumentParser()
    # Base directory to save downloaded files
    p.add_argument('-b', '--SAVEBASEDIR', dest='SAVEBASEDIR', type=str, 
                   default='./your_hpc_dir',
                   help='Remote directory to save downloaded files')
    # Year to download
    p.add_argument('-y', '--year', dest='year', type=str, default="2024",
                   help='Year for downloading')
    # Starting day (as string) -- will be converted to int for range creation
    p.add_argument('-d', '--day', dest='day', type=str, default="175",
                   help='Starting day as integer (e.g., "175")')
    # Number of consecutive days to download
    p.add_argument('-n', '--num_days', dest='num_days', type=int, default=10,
                   help='Number of consecutive days to download')
    # MODIS Base URL
    p.add_argument('-M', '--modis_basedir', dest='modis_basedir', type=str,
                   default='https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61',
                   help='Source directory to download datasets')
    # Config file path
    p.add_argument('-c', '--config_filepath', type=str,
                   default='./config.yml',
                   help='Path to config YAML file')
    FLAGS = p.parse_args()
    if verbose:
        for f in FLAGS.__dict__:
            print("\t", f, (25 - len(f)) * " ", FLAGS.__dict__[f])
        print("\n")
    return FLAGS


def laads_app_downloader(source, destination, token, app_logger, max_files=10):
    """
    Download files from a LAADS source directory to the destination,
    limiting the number of files downloaded to max_files.
    """
    import os
    import shutil
    import sys
    from io import StringIO
    import csv
    import json

    USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n', '').replace('\r', '')

    def getcURL(url, headers=None, out=None):
        import subprocess
        try:
            print('trying cURL', file=sys.stderr)
            args = ['curl', '--fail', '-sS', '-L', '-b', 'session', '--get', url]
            for (k, v) in headers.items():
                args.extend(['-H', f'{k}: {v}'])
            if out is None:
                result = subprocess.check_output(args)
                return result.decode('utf-8') if isinstance(result, bytes) else result
            else:
                subprocess.call(args, stdout=out)
        except subprocess.CalledProcessError as e:
            print('curl GET error message: %s' % (e.output if hasattr(e, 'output') else ''), file=sys.stderr)
        return None

    def geturl(url, token=None, out=None):
        headers = {'user-agent': USERAGENT}
        if token is not None:
            headers['Authorization'] = 'Bearer ' + token
        try:
            import ssl
            CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            if sys.version_info.major == 2:
                import urllib2
                try:
                    fh = urllib2.urlopen(urllib2.Request(url, headers=headers), context=CTX)
                    if out is None:
                        return fh.read()
                    else:
                        shutil.copyfileobj(fh, out)
                except urllib2.HTTPError as e:
                    print('TLSv1_2 sys 2 : HTTP GET error code: %d' % e.code, file=sys.stderr)
                    return getcURL(url, headers, out)
                except urllib2.URLError as e:
                    print('TLSv1_2 sys 2 : Failed to make request: %s, RETRYING' % e.reason, file=sys.stderr)
                    return getcURL(url, headers, out)
                return None
            else:
                from urllib.request import urlopen, Request, URLError, HTTPError
                try:
                    fh = urlopen(Request(url, headers=headers), context=CTX)
                    if out is None:
                        return fh.read().decode('utf-8')
                    else:
                        shutil.copyfileobj(fh, out)
                except HTTPError as e:
                    print('TLSv1_2 : HTTP GET error code: %d' % e.code, file=sys.stderr)
                    return getcURL(url, headers, out)
                except URLError as e:
                    print('TLSv1_2 : Failed to make request: %s' % e.reason, file=sys.stderr)
                    return getcURL(url, headers, out)
                return None
        except AttributeError:
            return getcURL(url, headers, out)

    def sync(src, dest, tok, max_files):
        """
        Synchronize the source URL with the destination directory.
        Downloads files (non-directories) until max_files have been downloaded.
        """
        try:
            # First try to get CSV listing; fall back to JSON if needed.
            csv_listing = geturl(f'{src}.csv', tok)
            files = {}
            files['content'] = [f for f in csv.DictReader(StringIO(csv_listing), skipinitialspace=True)]
        except Exception:
            json_listing = geturl(src + '.json', tok)
            files = json.loads(json_listing)

        count = 0
        for f in files['content']:
            # Expected filename structure: MOD09GA.AYYYYDDD.hXXvYY.061.YYYYMMDDHHMMSS.hdf
            parts = f['name'].split('.')
            if len(parts) < 3:
                app_logger.warning("Filename %s does not match expected format; skipping.", f['name'])
                continue
            tile = parts[2]
            # Allowed tile codes for Europe
            ALLOWED_TILES = {"h18v03", "h18v04", "h19v03", "h19v04"}
            if tile not in ALLOWED_TILES:
                app_logger.info("Skipping %s (tile %s not allowed).", f['name'], tile)
                continue
            
            if count >= max_files:
                break  # Only download up to max_files for this directory
            filesize = int(f['size'])
            path = os.path.join(dest, f['name'])
            url = src + '/' + f['name']
            if filesize == 0:  # Indicates a directory; recurse into it
                try:
                    print('creating dir:', path)
                    os.mkdir(path)
                    sync(src + '/' + f['name'], path, tok, max_files)
                except IOError as e:
                    print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                    sys.exit(-1)
            else:
                try:
                    if not os.path.exists(path) or os.path.getsize(path) == 0:
                        print('\ndownloading:', path)
                        with open(path, 'w+b') as fh:
                            geturl(url, tok, fh)
                        count += 1
                    else:
                        print('skipping:', path)
                except IOError as e:
                    print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                    sys.exit(-1)
        return 0

    if not os.path.exists(destination):
        try:
            os.makedirs(destination, exist_ok=True)
        except OSError as e:
            app_logger.error(f"Cannot create directory '{destination}': {e}")
            sys.exit(1)

    # Start the download process for this source directory
    r = sync(source, destination, token, max_files)
    return r


def prepare_paths(FLAGS, modis_basedir, BASEDIR, product, day):
    """
    Generate the source URL and destination directory for a given product,
    year, and day.
    """
    src = os.path.join(modis_basedir, product, FLAGS.year, day.zfill(3))
    savedir = os.path.join(BASEDIR, product, FLAGS.year, day.zfill(3))
    if not BASEDIR.startswith("/"):
        basedir = "./" + BASEDIR 
        savedir = os.path.join(basedir, product, FLAGS.year, day.zfill(3))
    elif BASEDIR.startswith("~"):
        basedir = os.path.expanduser("~") + BASEDIR[1:]
        savedir = os.path.join(basedir, product, FLAGS.year, day.zfill(3))
    return src, savedir


if __name__ == "__main__":

    FLAGS = get_args()

    # Load configuration
    with open(FLAGS.config_filepath, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            sys.exit(1)

    try:
        token = config['token']
    except KeyError:
        app_logger.error("Token missing in config file!")
        sys.exit(1)

    try:
        endpoint_id = config['machine1']['endpoint_id_name']
    except KeyError:
        app_logger.error("Endpoint ID missing in config file!")
        sys.exit(1)

    product = 'MOD09GA'
    start_day = int(FLAGS.day)
    num_days = FLAGS.num_days
    max_files = 10  # Maximum files to download per day

    download_gce = Executor(endpoint_id=endpoint_id)
    futures = []
    results = []

    # Loop over the specified day range
    for d in range(start_day, start_day + num_days):
        day_str = str(d).zfill(3)
        mod09GA_src, mod09GA_savedir = prepare_paths(FLAGS, FLAGS.modis_basedir, FLAGS.SAVEBASEDIR, product, day_str)
        print(f"Source URL for {product} on day {day_str}: {mod09GA_src}")
        print(f"Destination directory: {mod09GA_savedir}")
        try:
            app_logger.info(f"Submitting task for {product} download for day {day_str}...")
            future = download_gce.submit(laads_app_downloader, mod09GA_src, mod09GA_savedir, token, app_logger, max_files)
            futures.append(future)
            app_logger.info(f"Task submitted for day {day_str}")
        except Exception as e:
            app_logger.error(f"Error submitting task for day {day_str}: {e}")
            raise

    print("All tasks submitted; waiting for results.")

    # Optional: Check task status and retrieve results
    for future in futures:
        app_logger.info(f"Checking status of task: {future}")
        print("Task status:", future.done())

    for f in futures:
        try:
            app_logger.info(f"Retrieving result for task: {f}")
            r = f.result()  # This will raise an error if the task failed.
            results.append(r)
        except Exception as e:
            app_logger.error(f"Error getting result from task: {e}")
            results.append(None)
