import os
import bz2
import urllib

import matplotlib.pyplot as plt

def show_image(img):
    plt.imshow(img[:, :, ::-1])
    plt.show()

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
