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
    # imt1 = np.minimum(imdil, im)
    imt1 = cv2.min(imdil, im)
    is_equal = image_equal(imt0, imt1)
    #c = c + 1
  return imt1

def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

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
  nSteps = 100
  filename = "1_good.JPG"
  img = my.imread(filename)
  
  msg = ""
  if (TEST1):
    h = my.hist(img)
    my.showhist(h, 10)

  if (TEST2):
    my.imshow(my.histeq(imgray))

  if (TEST3):
    kernel_t = cv2.getGaussianKernel(51, 1)
    
    #print(kernel_t)
    kernel_size = 51
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    
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

    imdil_1 = cv2.dilate(imsmooth2, kernel_51, 1)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imdil_1 = cv2.dilate(imsmooth2, kernel_51, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Dilate: " + str(media) + " ms\n"
    total = total + media
    
    imdil_51 = cv2.dilate(imdil_1, kernel_1, 1)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imdil_51 = cv2.dilate(imdil_1, kernel_1, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Dilate: " + str(media) + " ms\n"
    total = total + media

    imerode_1 = cv2.erode(imdil_51, kernel_51, 1)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
     imerode_1 = cv2.erode(imdil_51, kernel_51, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Erode: " + str(media) + " ms\n"
    total = total + media

    
    imerode_51 = cv2.erode(imerode_1, kernel_1, 1)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imerode_51 = cv2.erode(imerode_1, kernel_1, 1)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Erode: " + str(media) + " ms\n"
    total = total + media
    

  

    result = imerode_51 - imsmooth2

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      result = imerode_51 - imsmooth2
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
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
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do método Threshold: " + str(media) + " ms\n"
    total = total + media

    #my.imshow(imthresh)

    #6 opening by reconstruction: erosão seguida da dilatação condicional até estabilização
    imopenrec = reconstruct(imthresh)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imopenrec = reconstruct(imthresh)
    t1 = datetime.now()
    t = t1 - t0
    media = round((t.total_seconds() * 1000) / nSteps, 2)
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do método Reconstruct: " + str(media) + " ms\n"
    total = total + media

    my.imshow(imopenrec)

# with open('PYTHON_TEST.txt', 'w') as arquivo:
#     print(msg)
#     print(msg, file=arquivo)
#     msg1 = "Valor total do tempo médio: "+str(total)
#     print(msg1, file=arquivo)
print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")
print("Valo total médio: "+ str(total))
