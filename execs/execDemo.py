# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import my
import cv2
import numpy as np
import time as t
from datetime import datetime

def close(im, kernel, iterations=1):
  imdil = cv2.dilate(im, kernel, iterations)
  result = cv2.erode(imdil, kernel, iterations)
  return result

def rgb2gray(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

def blackhat(im, kernel, iterations=1):
  result = close(im, kernel, iterations)
  return result

def smooth(im, diam=3):
  result = cv2.GaussianBlur(im, (diam, diam), 0)
  return(result)

def image_equal(im0, im1):
  return (sum(sum(im0 != im1)) == 0)


def reconstruct(im):
  kernel = np.ones((3, 3), np.uint8)
  imero =  cv2.erode(im, kernel)
  c = 0
  imt0 = imero
  imt1 = cv2.dilate(imt0, kernel)
  is_equal = image_equal(imt0, imt1)
  while (not is_equal):
    print(c)
    imt0 = imt1
    imdil = cv2.dilate(imt0, kernel)
    imt1 = np.minimum(imdil, im)
    is_equal = image_equal(imt0, imt1)
    c = c + 1
  return imt1


msg = ""
media = 0.0
nSteps = 10

TEST1 = False
TEST2 = False
TEST3 = True

if __name__ == "__main__":
  filename = "images/01.png"
  img = my.imread(filename)
  imgray = my.imreadgray(filename)
  

  if (TEST3):
    msg = ""
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = datetime.now()
    for i in range( nSteps ):
      imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do metódo Rgb2Gray: " + str(media) + " ms\n"
    #my.imshow(imgray)
    kernel = np.ones((5, 5), np.uint8)

    #1 Convolution
    imconv = smooth(imgray, 5)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imconv = smooth(imgray, 5)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do metódo Convolution: " + str(media) + " ms\n"

    #my.imshow(imconv)

    #2 Dilate    
    imdil = cv2.dilate(imconv, kernel, 1)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imdil = cv2.dilate(imconv, kernel, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do metódo Dilate: " + str(media) + " ms\n"

    #my.imshow(imdil)

    #3 Erode
    imerode = cv2.erode(imdil, kernel, 1)

    t0 = datetime.now()
    for i in range( nSteps ):
      imerode = cv2.erode(imdil, kernel, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do metódo Erode: " + str(media) + " ms\n"
    #my.imshow(imerode)


print("-------------------------------------------------------------")            
print(msg)
print("-------------------------------------------------------------")
