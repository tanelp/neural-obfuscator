import os
import bz2
import urllib
import html

import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(img):
    plt.imshow(img[:, :, ::-1])
    plt.show()

def url_to_img(url):
	req = urllib.request.urlopen(url)
	img = np.asarray(bytearray(req.read()), dtype=np.uint8)
	img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	return img

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, "wb") as fp:
        fp.write(data)
    return dst_path

def download_file(url, fname):
    datadir = os.path.expanduser(os.path.join("~", ".neural_obfuscator"))
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fpath = os.path.join(datadir, fname)

    if not os.path.exists(fpath):
        print("Downloading: " + url)
        urllib.request.urlretrieve(url, fpath)

    return fpath

def download_file_from_gdrive(url, outpath, num_attempts=5):
    url_data = None
    with requests.Session() as session:
        print("Downloading {} ...".format(url))
        for i in range(num_attempts):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive quota exceeded")

                    url_data = res.content
                    break
            except Exception as e:
                print("Attempt {}".format(i))
                print(e)
                if i >= num_attempts:
                    print("Download failed...")
                    raise

    # save
    if url_data is not None:
      print("Saving to {} ...".format(outpath))
      outdir = os.path.dirname(outpath)
      os.makedirs(outdir, exist_ok=True)
      with open(outpath, "wb") as f:
          f.write(url_data)
