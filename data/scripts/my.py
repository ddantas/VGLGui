import matplotlib.pyplot as mp
import numpy as np

import math as m
import time as t

from importlib import reload
#reload(my)

version = 3

def imread(filename):
    im = mp.imread(filename)
    if (im.dtype == "float32"):
        im = np.uint8(255*im)
    if (nchannels(im) > 3):
        im = im[:, :, 0:3]
    return im

def imreadgray(filename):
    im = imread(filename)
    if (nchannels(im) == 1):
        return im
    else:
        return rgb2gray(im)

def rgb2gray(im):
    result = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    result[result > 255] = 255
    np.round(result)
    return np.uint8(result)

def imshow(im):
    plot = mp.imshow(im, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
    plot.set_interpolation('nearest')
    mp.show()

def nchannels(im):
    if (len(im.shape) >= 3):
        return im.shape[2]
    else:
        return 1

def thresh(im, m):
    result = im >= m
    result = np.uint8(result * 255)
    return result

def hist(im):
    c = nchannels(im)
    result = np.zeros((256, c), dtype=np.uint32)
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            if (c == 1):
                result[im[i, j]] +=1
            else:
                for k in range(0, c):
                    result[im[i, j, k], k] +=1
    return result

def showhist(h, bins=1):
    cols = h.shape[1]
    lins = h.shape[0]
    lins2 = (lins-1)//bins+1
    h2 = np.zeros(( lins2, cols ))
    for j in range(cols):
        for i in range(lins):
            i2 = i//bins
            h2[i2, j] += h[i, j]

    print(h2)

    if (h.shape[1] == 1):
        mp.bar(range(0, lins2), h, color='gray')
        mp.show()
    else:
        mp.bar(np.arange(lins2),       h2[:, 2], 1./3., color='blue',  edgecolor='none')
        mp.bar(np.arange(lins2)+1./3., h2[:, 1], 1./3., color='green', edgecolor='none')
        mp.bar(np.arange(lins2)+2./3., h2[:, 0], 1./3., color='red',   edgecolor='none')
        mp.show()

def histeq(im):
    if (nchannels(im) != 1):
        print("Error: histeq: only grayscale images are supported")
        return
    h = hist(im)
    n = np.sum(h)
    prob = np.zeros(256)
    for i in range(256):
        if (i == 0):
            prob[0] = h[0, 0]
        else:
            prob[i] = h[i, 0] + prob[i-1]
    t = (prob / n) * 255.0
    result = 0.0 * im
    for i in range(256):
        result[im == i] = t[i]
    result = np.uint8(result)
    return result

# mask333 = [1.0/27.0 for _ in range(27)]
# print(mask333)



erode_mask = [1] * 625
print(erode_mask)
