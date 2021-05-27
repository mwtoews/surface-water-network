#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download USGS executables, and install them in bindir
"""

import os
import platform
import sys
import tempfile
import urllib
import zipfile

import requests

# See https://docs.github.com/en/rest/reference/repos#releases
owner = "MODFLOW-USGS"
repo = "executables"
api_url = "https://api.github.com/repos/{}/{}".format(owner, repo)


def main(release_id="latest", bindir="bin"):
    """Run main component of script."""
    os.makedirs(bindir, exist_ok=True)
    req_url = api_url + "/releases/" + release_id
    resp = requests.get(req_url)
    if not resp.ok:
        raise RuntimeError("{}: {}".format(req_url, str(resp)))
    release = resp.json()
    assets = release.get("assets", [])
    print("fetched release {} with {} assets".format(
        release["tag_name"], len(assets)))
    if sys.platform.startswith("linux"):
        ostag = "linux"
    elif sys.platform.startswith("win"):
        ostag = "win" + platform.architecture()[0][:2]
    elif sys.platform.startswith("darwin"):
        ostag = "mac"
    found_ostag = False
    for asset in assets:
        if ostag in asset["name"]:
            found_ostag = True
            break
    if not found_ostag:
        raise KeyError(
            "could not find {} from {}; see available assets here: {}".format(
                ostag, release["tag_name"], release["html_url"]))
    with tempfile.TemporaryDirectory() as dname:
        download_pth = os.path.join(dname, asset["name"])
        print("downloading {}".format(asset["name"]))
        urllib.request.urlretrieve(asset["browser_download_url"], download_pth)
        print("extracting files to {}".format(os.path.abspath(bindir)))
        file_list = []
        with zipfile.ZipFile(download_pth, "r") as zf:
            for file in zf.filelist:
                name = file.filename
                perm = ((file.external_attr >> 16) & 0o777)
                file_list.append(name)
                if name.endswith("/"):
                    os.mkdir(os.path.join(bindir, name), perm)
                else:
                    outfile = os.path.join(bindir, name)
                    fh = os.open(outfile, os.O_CREAT | os.O_WRONLY, perm)
                    os.write(fh, zf.read(name))
                    os.close(fh)
        print("extracted {} files: {}".format(
            len(file_list), ", ".join(file_list)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release_id", default="latest",
        help="release_id (default: %(default)s)")
    parser.add_argument(
        "--bindir", default="bin",
        help="directory to put executables (default: %(default)s)")
    args = parser.parse_args()
    main(**vars(args))
