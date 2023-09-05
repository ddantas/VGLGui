# -*- coding: utf-8 -*-
#!/usr/bin/env python2
import my
import cv2
import numpy as np
import time as t
from datetime import datetime

def close(im, kernel, iterations=1):
  imdil = cv2.dilate(im, kernel, iterations)
  result = cv2.erode(imdil, kernel, iterations)
  return result

def blackhat(im, kernel, iterations=1):
  result = close(im, kernel, iterations)
  return result

def smooth(im, diam=3):
  result = cv2.GaussianBlur(im, (diam, diam), 0)
  return(result)

def image_equal(im0, im1):
  return (sum(sum(im0 != im1)) == 0)


def reconstruct(im):
  kernel = np.ones((17, 17), np.uint8)
  imero =  cv2.erode(im, kernel)
  c = 0
  imt0 = imero
  imt1 = cv2.dilate(imt0, kernel)
  is_equal = image_equal(imt0, imt1)
  while (not is_equal.all()):
    #print(c)
    imt0 = imt1
    imdil = cv2.dilate(imt0, kernel)
    imt1 = np.minimum(imdil, im)
    is_equal = image_equal(imt0, imt1)
    c = c + 1
  return imt1

def tratnum (num):
    listnum = []
    for line in num:
        listnum.append(float(line))
        listnumpy = np.array(listnum,np.float32)
    return listnumpy

TEST1 = False
TEST2 = False
TEST3 = True
total = 0.0
msg = ""
if __name__ == "__main__":
  nSteps = 10
  filename = "images/1_good.jpg"
  img = my.imread(filename)
  
  msg = ""
  if (TEST3):
    kernel_t = cv2.getGaussianKernel(51, 1)
    
    #print(kernel_t)
    kernel_size = 51
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    #print(gauss)
    
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #my.imshow(imgray)
    t0 = datetime.now()
    for i in range( nSteps ):
      imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Rgb2Gray: " + str(media) + " ms\n"
    total = total + media

    #2 suavização
    #imsmooth1 = smooth(imgray, 15)
    imsmooth = cv2.sepFilter2D(imgray,-1,gauss,gauss)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imsmooth = cv2.sepFilter2D(imgray,-1,gauss,gauss)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Convolution: " + str(media) + " ms\n"
    total = total + media
    my.imshow(imsmooth)

   
   
    
    #y.imshow(imsmooth)
    kernel_51 = np.ones((51, 1), np.uint8)
    kernel_1 = np.ones((1, 51), np.uint8)

    imdil_1 = cv2.dilate(imsmooth, kernel_51, 1)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imdil_1 = cv2.dilate(imsmooth, kernel_51, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Dilate: " + str(media) + " ms\n"
    total = total + media
    
    imdil_51 = cv2.dilate(imdil_1, kernel_1, 1)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imdil_51 = cv2.dilate(imdil_1, kernel_1, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Dilate: " + str(media) + " ms\n"
    total = total + media

    imerode_1 = cv2.erode(imdil_51, kernel_51, 1)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
     imerode_1 = cv2.erode(imdil_51, kernel_51, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Erode: " + str(media) + " ms\n"
    total = total + media

    
    imerode_51 = cv2.erode(imerode_1, kernel_1, 1)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imerode_51 = cv2.erode(imerode_1, kernel_1, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Erode: " + str(media) + " ms\n"
    total = total + media
    

  

    result = imerode_51 - imsmooth

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      result = imerode_51 - imsmooth
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do método Sub: " + str(media) + " ms\n"
    total = total + media

    #my.imshow(my.histeq(result))

    #5 threshold
    imthresh = my.thresh(result, 2)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imthresh = my.thresh(result, 2)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do método Threshold: " + str(media) + " ms\n"
    total = total + media

    #my.imshow(imthresh)

    #6 opening by reconstruction: erosão seguida da dilatação condicional até estabilização
    imopenrec = reconstruct(imthresh)
    my.imshow(imopenrec)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imopenrec = reconstruct(imthresh)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do método Reconstruct: " + str(media) + " ms\n"
    total = total + media



with open('files/PYTHON_TEST_NEW.txt', 'w') as arquivo:
    print(msg)
    print(msg, file=arquivo)
    msg1 = "Valor total do tempo médio: "+str(total)
    print(msg1, file=arquivo)
print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")
print("Valo total médio: "+ str(total))
