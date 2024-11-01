#!/usr/bin/env python3

from vgl_lib.vglClUtil import vglClEqual

from vgl_lib.vglImage import VglImage
import pyopencl as cl       # OPENCL LIBRARY
import vgl_lib as vl        # VGL LIBRARYS
import numpy as np          # TO WORK WITH MAIN
from cl2py_shaders import * # IMPORTING METHODS
from cl2py_ND import *
import os
import sys                  # IMPORTING METHODS FROM VGLGui
from readWorkflow import *
import time as t
from datetime import datetime

import matplotlib.pyplot as mp


os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
sys.path.append(os.getcwd())

# Actions after glyph execution
def GlyphExecutedUpdate(GlyphExecutedUpdate_Glyph_Id, GlyphExecutedUpdate_image):

    # Rule10: Glyph becomes DONE = TRUE after its execution. Assign done to glyph
    setGlyphDoneId(GlyphExecutedUpdate_Glyph_Id)

    # Rule6: Edges whose source glyph has already been executed, and which therefore already had their image generated, have READY=TRUE (image ready to be processed).
    #        Reading the image from another glyph does not change this status. Check the list of connections
    setGlyphInputReadyByIdOut(GlyphExecutedUpdate_Glyph_Id) 

    # Rule2: In a source glyph, images (one or more) can only be output parameters.
    setImageConnectionByOutputId(GlyphExecutedUpdate_Glyph_Id, GlyphExecutedUpdate_image)
                
# Program execution

# Reading the workflow file and loads into memory all glyphs and connections
# Rule7: Glyphs have READY (ready to run) and DONE (executed) status, both status start being FALSE
fileRead(lstGlyph, lstConnection)

def imshow(im):
    plot = mp.imshow(im, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
    plot.set_interpolation('nearest')
    mp.show()

def tratnum (num):
    listnum = []
    for line in num:
        listnum.append(float(line))
        listnumpy = np.array(listnum, np.float32)
    return listnumpy

#nSteps = int(sys.argv[2])
nSteps = 1
msg = ""
CPU = cl.device_type.CPU #2
GPU = cl.device_type.GPU #4
total = 0.0
vl.vglClInit(GPU) 

# Update the status of glyph entries
for vGlyph in lstGlyph:
    
    # Rule9: Glyphs whose status is READY=TRUE (ready to run) are executed. Only run the glyph if all its entries are
    try:
        if not vGlyph.getGlyphReady():
            raise Error("Rule9: Glyph not ready for processing.", {vGlyph.glyph_id})
    except ValueError:
        print("Rule9: Glyph not ready for processing: ", {vGlyph.glyph_id})

    if vGlyph.func == 'vglLoadImage':

        vglLoadImage_img_in_path = vGlyph.lst_par[0].getValue()
        
        vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
        #vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_2D_IMAGE(), vl.IMAGE_ND_ARRAY())
        vl.vglLoadImage(vglLoadImage_img_input)
        if( vglLoadImage_img_input.getVglShape().getNChannels() == 3 ):
          vl.rgb_to_rgba(vglLoadImage_img_input)

        vl.vglClUpload(vglLoadImage_img_input)

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglLoadImage_img_input)


    elif vGlyph.func == 'vglClRgb2Gray': #Function Rgb2Gray
        print("-------------------------------------------------")
        print("A função " + vGlyph.func +" está sendo executada")
        print("-------------------------------------------------")
    
        # Search the input image by connecting to the source glyph
        vglClRgb2Gray_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

        # Search the output image by connecting to the source glyph
        vglClRgb2Gray_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        # Apply SwapRgb function
        vglClRgb2Gray(vglClRgb2Gray_img_input ,vglClRgb2Gray_img_output)

        #Runtime
        vl.get_ocl().commandQueue.flush()
        t0 = datetime.now()
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClRgb2Gray_img_output)


    elif vGlyph.func == 'vglCreateImage':

        # Search the input image by connecting to the source glyph
        vglCreateImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img')

        vglCreateImage_RETVAL = vl.create_blank_image_as(vglCreateImage_img_input)
        vglCreateImage_RETVAL.set_oclPtr( vl.get_similar_oclPtr_object(vglCreateImage_img_input) )
        vl.vglAddContext(vglCreateImage_RETVAL, vl.VGL_CL_CONTEXT())

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglCreateImage_RETVAL)

    elif vGlyph.func == 'Reconstruct': #Function Reconstruct
        print("-------------------------------------------------")
        print("A função " + vGlyph.func +" está sendo executada")
        print("-------------------------------------------------")
    
        # Search the input image by connecting to the source glyph
        Rec_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

        

        # Search the output image by connecting to the source glyph
        Rec_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        n_pixel = np.uint32(vGlyph.lst_par[0].getValue())
        elemento = tratnum(vGlyph.lst_par[0].getValue())
        x = np.uint32(vGlyph.lst_par[1].getValue())
        y = np.uint32(vGlyph.lst_par[2].getValue())


        #Runtime
        vl.get_ocl().commandQueue.flush()
        t0 = datetime.now()
        Rec_imt1 = vl.create_blank_image_as(Rec_img_input)
        Rec_buffer = vl.create_blank_image_as(Rec_img_input)
        for i in range( nSteps ):
          
          vglClErode(Rec_img_input, Rec_img_output, elemento, x, y)

          result = 0
          count = 0
          while (not result ):
            if ((count % 2) == 0):
              vglClDilate( Rec_img_output , Rec_buffer ,elemento, x, y)
              vglClMin(Rec_buffer , Rec_img_input, Rec_imt1)
            else:
              vglClDilate( Rec_imt1 , Rec_buffer , elemento, x, y)
              vglClMin(Rec_buffer, Rec_img_input, Rec_img_output)
            result = vglClEqual(Rec_imt1, Rec_img_output)
            count = count + 1
          
          #print("contador reconstrcut",count)  

        vl.get_ocl().commandQueue.finish()
        t1 = datetime.now()
        diff = t1 - t0
        media = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Reconstruct: " + str(media) + " ms\n"
        total = total + media

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id,Rec_img_output)

    elif vGlyph.func == 'vglSaveImage':

        # Returns edge image based on glyph id
        vglSaveImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'image')

        if vglSaveImage_img_input is not None:

            # SAVING IMAGE img
            vpath = vGlyph.lst_par[0].getValue()

            # Rule3: In a sink glyph, images (one or more) can only be input parameters
            vl.vglCheckContext(vglSaveImage_img_input,vl.VGL_RAM_CONTEXT())             
            vl.vglSaveImage(vpath, vglSaveImage_img_input)
            

            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, None)

    elif vGlyph.func == 'vglCl3dBlurSq3':

        vglCl3dBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglCl3dBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dBlurSq3(vglCl3dBlurSq3_img_input, vglCl3dBlurSq3_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dBlurSq3_img_output)


    elif vGlyph.func == 'vglCl3dConvolution':

        vglCl3dConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglCl3dConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dConvolution(vglCl3dConvolution_img_input, vglCl3dConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dConvolution_img_output)


    elif vGlyph.func == 'vglCl3dCopy':

        vglCl3dCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglCl3dCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dCopy(vglCl3dCopy_img_input, vglCl3dCopy_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dCopy_img_output)


    elif vGlyph.func == 'vglCl3dDilate':

        vglCl3dDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglCl3dDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dDilate(vglCl3dDilate_img_input, vglCl3dDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dDilate_img_output)


    elif vGlyph.func == 'vglCl3dErode':

        vglCl3dErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglCl3dErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dErode(vglCl3dErode_img_input, vglCl3dErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dErode_img_output)


    elif vGlyph.func == 'vglCl3dMax':

        vglCl3dMax_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
        vglCl3dMax_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
        vglCl3dMax_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dMax(vglCl3dMax_img_input1, vglCl3dMax_img_input2, vglCl3dMax_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dMax_img_output)


    elif vGlyph.func == 'vglCl3dMin':

        vglCl3dMin_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
        vglCl3dMin_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
        vglCl3dMin_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dMin(vglCl3dMin_img_input1, vglCl3dMin_img_input2, vglCl3dMin_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dMin_img_output)


    elif vGlyph.func == 'vglCl3dNot':

        vglCl3dNot_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglCl3dNot_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dNot(vglCl3dNot_img_input, vglCl3dNot_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dNot_img_output)


    elif vGlyph.func == 'vglCl3dSub':

        vglCl3dSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
        vglCl3dSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
        vglCl3dSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dSub(vglCl3dSub_img_input1, vglCl3dSub_img_input2, vglCl3dSub_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dSub_img_output)


    elif vGlyph.func == 'vglCl3dSum':

        vglCl3dSum_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
        vglCl3dSum_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
        vglCl3dSum_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglCl3dSum(vglCl3dSum_img_input1, vglCl3dSum_img_input2, vglCl3dSum_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dSum_img_output)


    elif vGlyph.func == 'vglCl3dThreshold':

        vglCl3dThreshold_src = getImageInputByIdName(vGlyph.glyph_id, 'src')
        vglCl3dThreshold_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst')
        vglCl3dThreshold(vglCl3dThreshold_src, vglCl3dThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))

        GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dThreshold_dst)


    elif vGlyph.func == 'vglClBlurSq3':

        vglClBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglClBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClBlurSq3(vglClBlurSq3_img_input, vglClBlurSq3_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClBlurSq3_img_output)


    elif vGlyph.func == 'vglClConvolution':

        vglClConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglClConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClConvolution(vglClConvolution_img_input, vglClConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClConvolution_img_output)


    elif vGlyph.func == 'vglClCopy':

        vglClCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglClCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClCopy(vglClCopy_img_input, vglClCopy_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClCopy_img_output)


    elif vGlyph.func == 'vglClDilate':

        vglClDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglClDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClDilate(vglClDilate_img_input, vglClDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClDilate_img_output)


    elif vGlyph.func == 'vglClErode':

        vglClErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglClErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClErode(vglClErode_img_input, vglClErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClErode_img_output)


    elif vGlyph.func == 'vglClInvert':

        vglClInvert_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglClInvert_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClInvert(vglClInvert_img_input, vglClInvert_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClInvert_img_output)


    elif vGlyph.func == 'vglClMax':

        vglClMax_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
        vglClMax_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
        vglClMax_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClMax(vglClMax_img_input1, vglClMax_img_input2, vglClMax_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClMax_img_output)


    elif vGlyph.func == 'vglClMin':

        vglClMin_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
        vglClMin_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
        vglClMin_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClMin(vglClMin_img_input1, vglClMin_img_input2, vglClMin_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClMin_img_output)


    elif vGlyph.func == 'vglClRgb2Gray':

        vglClRgb2Gray_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vglClRgb2Gray_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClRgb2Gray(vglClRgb2Gray_img_input, vglClRgb2Gray_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClRgb2Gray_img_output)


    elif vGlyph.func == 'vglClSub':

        vglClSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
        vglClSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
        vglClSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClSub(vglClSub_img_input1, vglClSub_img_input2, vglClSub_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClSub_img_output)


    elif vGlyph.func == 'vglClSum':

        vglClSum_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
        vglClSum_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
        vglClSum_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vglClSum(vglClSum_img_input1, vglClSum_img_input2, vglClSum_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClSum_img_output)


    elif vGlyph.func == 'vglClSwapRgb':

        vglClSwapRgb_src = getImageInputByIdName(vGlyph.glyph_id, 'src')
        vglClSwapRgb_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst')
        vglClSwapRgb(vglClSwapRgb_src, vglClSwapRgb_dst)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClSwapRgb_dst)


    elif vGlyph.func == 'vglClThreshold':

        vglClThreshold_src = getImageInputByIdName(vGlyph.glyph_id, 'src')
        vglClThreshold_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst')
        vglClThreshold(vglClThreshold_src, vglClThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClThreshold_dst)

