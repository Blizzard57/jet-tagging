import requests
import pathlib
import tqdm
import functools
import shutil
import os
import logging


def download(url, filename, exists_ok=True):
    if os.path.exists(filename) and exists_ok:
        logging.info("File already exists: %s" % filename)
        return
    else:
        logging.info("Downloading the file: %s" % filename)
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.tqdm.wrapattr(r.raw, "read", total=file_size) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path
