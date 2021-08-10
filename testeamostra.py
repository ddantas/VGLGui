#!/usr/bin/env python3

# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_shaders import * 

import time as t
import sys

"""
	THIS BENCHMARK TOOL EXPECTS 2 ARGUMENTS:
	ARGV[1]: PRIMARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
		IT WILL BE USED IN ALL KERNELS
	ARGV[2]: SECONDARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
		IT WILL BE USED IN THE KERNELS THAT NEED
		TWO INPUT IMAGES TO WORK PROPERLY
	THE RESULT IMAGES WILL BE SAVED AS IMG-[PROCESSNAME].JPG
"""


def salvando2d(img, name):
	# SAVING IMAGE img
	ext = name.split(".")
	ext.reverse()

	#vl.vglClDownload(img)
	vl.vglCheckContext(img, vl.VGL_RAM_CONTEXT())

	if( ext.pop(0).lower() == 'jpg' ):
		if( img.getVglShape().getNChannels() == 4 ):
			vl.rgba_to_rgb(img)
	
	vl.vglSaveImage(name, img)


img_in_path = sys.argv[1]
nSteps		= 100
img_out_path= sys.argv[2]

msg = ""

vl.vglClInit()

img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input)
if( img_input.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input)

vl.vglClUpload(img_input)

img_output = vl.create_blank_image_as(img_input)
img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())





#cv = np.array((1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9),np.float32) # Filtro média
#cv = np.array((1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3),np.float32)  # Filtro média
#cv = np.array(	(1, 1, 1, 1, 1, 1, 1, 1, 1 ), np.float32)
cv = np.array(	(0, 1, 0, 1, 1, 1, 0, 1, 0 ), np.float32)

print(cv)

#vglClConvolution(img_input, img_output, cv, np.uint32(3), np.uint32(3))

#salvando2d(img_output, img_out_path+"img-vglClConvolution.jpg")
#vl.rgb_to_rgba(img_output)



vglClErode(img_input,img_output, cv, np.uint32(3), np.uint32(3))

salvando2d(img_output, img_out_path+"img-vglClErodeCross1.png")
vl.rgb_to_rgba(img_output)








