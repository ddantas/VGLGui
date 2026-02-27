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
    # print(c)
    imt0 = imt1
    imdil = cv2.dilate(imt0, kernel)
    imt1 = np.minimum(imdil, im)
    is_equal = image_equal(imt0, imt1)
    c = c + 1
  return imt1


msg = ""
media = 0.0
nSteps = 100
total = 0.0
TEST1 = False
TEST2 = False
TEST3 = True

if __name__ == "__main__":
  filename = "1_good.JPG"
  img = my.imread(filename)
  imgray = my.imreadgray(filename)
  

  if (TEST3):
    msg = ""
    kernel_t = cv2.getGaussianKernel(51, 1)
    
    #print(kernel_t)
    kernel_size = 51
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    # print(gauss)
    
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #my.imshow(imgray)
    t0 = datetime.now()
    for i in range( nSteps ):
      imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Rgb2Gray: " + str(media) + " ms\n"
    total = total + media

    #2 suavização
    #imsmooth1 = smooth(imgray, 15)
    imsmooth = cv2.sepFilter2D(imgray,-1,gauss,gauss)
    #imsmooth1 = convolve(imgray, tratnum(kernel_15x15))

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imsmooth = cv2.sepFilter2D(imgray,-1,gauss,gauss)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Convolution: " + str(media) + " ms\n"
    total = total + media

    imsmooth2 = cv2.sepFilter2D(imsmooth,-1,gauss,gauss)
    
    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imsmooth2 = cv2.sepFilter2D(imsmooth,-1,gauss,gauss)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Convolution: " + str(media) + " ms\n"
    total = total + media
    
    #my.imshow(imsmooth)
    kernel_51 = np.ones((51, 1), np.uint8)
    kernel_1 = np.ones((1, 51), np.uint8)
    kernel_3 = np.ones((3,3), np.uint8)
    imdil_1 = cv2.dilate(imsmooth2, kernel_3, kernel_3)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imdil_1 = cv2.dilate(imsmooth2, kernel_3,kernel_3)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Dilate: " + str(media) + " ms\n"
    total = total + media
    
    imdil_51 = cv2.dilate(imdil_1, kernel_3, kernel_3)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imdil_51 = cv2.dilate(imdil_1, kernel_3, kernel_3)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Dilate: " + str(media) + " ms\n"
    total = total + media

    imerode_1 = cv2.erode(imdil_51, kernel_3, kernel_3)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
     imerode_1 = cv2.erode(imdil_51, kernel_3, kernel_3)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Erode: " + str(media) + " ms\n"
    total = total + media

    
    imerode_51 = cv2.erode(imerode_1, kernel_3, kernel_3)
    # imshow(imerode_51)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imerode_51 = cv2.erode(imerode_1, kernel_3, kernel_3)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Erode: " + str(media) + " ms\n"
    total = total + media

print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")
print("Valo total médio: "+ str(total))
