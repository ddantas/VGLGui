#!/usr/bin/env python3

from vglClUtil import vglClEqual
from vgl_lib import vglClImage
from vgl_lib.vglImage import VglImage
import pyopencl as cl       # OPENCL LIBRARY
import vgl_lib as vl        # VGL LIBRARYS
import numpy as np          # TO WORK WITH MAIN
from cl2py_shaders import * # IMPORTING METHODS
import os
import sys                  # IMPORTING METHODS FROM VGLGui
from readWorkflow import *
import time as t
import gc
from datetime import datetime

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

import matplotlib.pyplot as mp

def imshow(im):
    plot = mp.imshow(im, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
    plot.set_interpolation('nearest')
    mp.show()

def tratnum (num):
    listnum = []
    for line in num:
        listnum.append(float(line))
        listnumpy = np.array(listnum,np.float32)
    return listnumpy

nSteps = int(sys.argv[2])
msg = ""
CPU = cl.device_type.CPU #2
GPU = cl.device_type.GPU #4
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

        # Read "-filename" entry from glyph vglLoadImage
        vglLoadImage_img_in_path = vGlyph.lst_par[0].getValue()               
        vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())

        vl.vglLoadImage(vglLoadImage_img_input)
        if( vglLoadImage_img_input.getVglShape().getNChannels() == 3 ):
            vl.rgb_to_rgba(vglLoadImage_img_input)

        vl.vglClUpload(vglLoadImage_img_input)

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglLoadImage_img_input)
                                
    elif vGlyph.func == 'vglCreateImage':

        # Search the input image by connecting to the source glyph
        vglCreateImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img')

        vglCreateImage_RETVAL = vl.create_blank_image_as(vglCreateImage_img_input)
        vglCreateImage_RETVAL.set_oclPtr( vl.get_similar_oclPtr_object(vglCreateImage_img_input) )
        vl.vglAddContext(vglCreateImage_RETVAL, vl.VGL_CL_CONTEXT())

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglCreateImage_RETVAL)

    elif vGlyph.func == 'vglClBlurSq3': #Function blur

        # Search the input image by connecting to the source glyph
        vglClBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        
        # Search the output image by connecting to the source glyph
        vglClBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        # Apply BlurSq3 function
        vglClBlurSq3(vglClBlurSq3_img_input, vglClBlurSq3_img_output)

        #Runtime
        t0 = datetime.now()
        for i in range( nSteps ):
          vglClBlurSq3(vglClBlurSq3_img_input, vglClBlurSq3_img_output)
        t1 = datetime.now()
        t = t1 - t0
        media = (t.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClBlurSq3: " + str(media) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClBlurSq3_img_output)


    elif vGlyph.func == 'vglClErode': #Function Erode

        # Search the input image by connecting to the source glyph
        vglClErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        
        # Search the output image by connecting to the source glyph
        vglClErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
       
        # Apply Erode function
        vl.vglCheckContext(vglClErode_img_output,vl.VGL_CL_CONTEXT())
        vglClErode(vglClErode_img_input, vglClErode_img_output, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
        
        #Runtime
        t0 = datetime.now()
        for i in range( nSteps ):
          vglClErode(vglClErode_img_input, vglClErode_img_output, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        t1 = datetime.now()
        t = t1 - t0
        media = (t.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClErode: " + str(media) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClErode_img_output)

    elif vGlyph.func == 'vglClConvolution': #Function Convolution

        # Search the input image by connecting to the source glyph
        vglClConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        
        # Search the output image by connecting to the source glyph
        vglClConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        # Apply Convolution function
        #vl.vglCheckContext(vglClConvolution_img_output,vl.VGL_CL_CONTEXT())
        vglClConvolution(vglClConvolution_img_input, vglClConvolution_img_output,tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        #Runtime
        t0 = datetime.now()
        for i in range( nSteps ):
          vglClConvolution(vglClConvolution_img_input, vglClConvolution_img_output,tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClConvolution: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClConvolution_img_output)

    elif vGlyph.func == 'vglClDilate': #Function Dilate
    
        # Search the input image by connecting to the source glyph
        vglClDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

        # Search the output image by connecting to the source glyph
        vglClDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        # Apply Dilate function
        vl.vglCheckContext(vglClDilate_img_output,vl.VGL_CL_CONTEXT())
        vglClDilate(vglClDilate_img_input, vglClDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        #Runtime
        t0 = datetime.now()

        for i in range( nSteps ):
          vglClDilate(vglClDilate_img_input, vglClDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClDilate: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClDilate_img_output)

    elif vGlyph.func == 'vglClThreshold': #Function Threshold
    
        # Search the input image by connecting to the source glyph
        vglClThreshold_img_input = getImageInputByIdName(vGlyph.glyph_id, 'src')

        # Search the output image by connecting to the source glyph
        vglClThreshold_img_output = getImageInputByIdName(vGlyph.glyph_id, 'dst')

        # Apply Threshold function
        vglClThreshold(vglClThreshold_img_input, vglClThreshold_img_output, np.float32(vGlyph.lst_par[0].getValue()))

        #Runtime
        t0 = datetime.now()

        for i in range( nSteps ):
          vglClThreshold(vglClThreshold_img_input, vglClThreshold_img_output, np.float32(vGlyph.lst_par[0].getValue()))
        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClThreshold: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClThreshold_img_output)
    
    elif vGlyph.func == 'vglClSwapRgb': #Function SwapRGB
    
        # Search the input image by connecting to the source glyph
        vglClSwapRgb_img_input = getImageInputByIdName(vGlyph.glyph_id, 'src')

        # Search the output image by connecting to the source glyph
        vglClSwapRgb_img_output = getImageInputByIdName(vGlyph.glyph_id, 'dst')

        # Apply SwapRgb function
        vglClSwapRgb(vglClSwapRgb_img_input,vglClSwapRgb_img_output)

        #Runtime
        t0 = datetime.now()

        for i in range( nSteps ):
          vglClSwapRgb(vglClSwapRgb_img_input,vglClSwapRgb_img_output)
        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClSwapRgb: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClSwapRgb_img_output)


    elif vGlyph.func == 'vglClRgb2Gray': #Function SwapRGB
    
        # Search the input image by connecting to the source glyph
        vglClRgb2Gray_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

        # Search the output image by connecting to the source glyph
        vglClRgb2Gray_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        # Apply SwapRgb function
        vglClRgb2Gray(vglClRgb2Gray_img_input ,vglClRgb2Gray_img_output)

        #Runtime
        t0 = datetime.now()

        for i in range( nSteps ):
          vglClRgb2Gray(vglClRgb2Gray_img_input ,vglClRgb2Gray_img_output)
        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClRgb2Gray: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClRgb2Gray_img_output)
    
    elif vGlyph.func == 'vglClInvert': #Function Invert

        # Search the input image by connecting to the source glyph
        vglClInvert_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        
        # Search the output image by connecting to the source glyph
        vglClInvert_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        # Apply Invert function
        vglClInvert(vglClInvert_img_input, vglClInvert_img_output)

        #Runtime
        t0 = datetime.now()
        for i in range( nSteps ):
          vglClInvert(vglClInvert_img_input, vglClInvert_img_output)
        
        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClInvert: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClInvert_img_output)

    elif vGlyph.func == 'vglClSub': #Function Sub

        # Search the input image by connecting to the source glyph
        vglClSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
        
        # Search the output image by connecting to the source glyph
        
        vglClSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        vglClSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')

        # Apply Sub Function
        #vl.vglCheckContext(vglClSub_img_input1 ,vl.VGL_RAM_CONTEXT())
        #vl.vglCheckContext(vglClSub_img_input2 ,vl.VGL_RAM_CONTEXT())
        #vl.vglCheckContext(vglClSub_img_output ,vl.VGL_RAM_CONTEXT())
        
        vglClSub(vglClSub_img_input1,vglClSub_img_input2,vglClSub_img_output)

        #Runtime
        t0 = datetime.now()
        for i in range( nSteps ):
          vglClSub(vglClSub_img_input1,vglClSub_img_input2,vglClSub_img_output)

        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClSub: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClSub_img_output)


    elif vGlyph.func == 'vglClMin': #Function Min

        # Search the input image by connecting to the source glyph
        vglClMin_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        
        # Search the output image by connecting to the source glyph
        vglClMin_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        
        
        # Apply Min function
        vglClMin(vglClMin_img_input, vglClMin_img_output,vglClMin_img_output  )

        #Runtime
        t0 = datetime.now()
        for i in range( nSteps ):
          vglClMin(vglClMin_img_input, vglClMin_img_output,vglClMin_img_output  )

        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClMin: " + str(med) + " ms\n"
        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClMin_img_output)

    elif vGlyph.func == 'vglClSum': #Function Sum
    
        # Search the input image by connecting to the source glyph
        vglClSum_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

        # Search the output image by connecting to the source glyph
        vglClSum_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        # Apply Sumfunction
        vglClSum(vglClSum_img_input,vglClSum_img_output,vglClSum_img_output) 

        #Runtime
        t0 = datetime.now()
        for i in range( nSteps ):
          vglClSum(vglClSum_img_input,vglClSum_img_output,vglClSum_img_output)

        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClSum: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, vglClSum_img_output)

    elif vGlyph.func == 'Closing': #Function Closing

        # Search the input image by connecting to the source glyph
        Closing_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

        # Search the output image by connecting to the source glyph
        Closing_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        
        Closing_buffer = vl.create_blank_image_as(Closing_img_input)

        vglClDilate(Closing_img_input, Closing_buffer, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        vglClErode(Closing_buffer, Closing_img_output , tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        #gc.collect(Closing_buffer)

        #Runtime
        t0 = datetime.now()
        for i in range( nSteps ):
          vglClDilate(Closing_img_input, Closing_buffer, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          vglClErode(Closing_buffer, Closing_img_output , tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo Closing: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, Closing_img_output)


    elif vGlyph.func == 'Reconstruct': #Function Reconstruct
    
        # Search the input image by connecting to the source glyph
        Rec_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

        # Search the output image by connecting to the source glyph
        Rec_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

        Rec_buffer = vl.create_blank_image_as(Rec_img_input)
        
        vglClErode(Rec_img_input, Rec_buffer, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
        #gc.collect(Rec_buffer)
        #vglClDilate(Rec_buffer, Rec_img_output, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
        
        result = vglClEqual(Rec_buffer,Rec_img_input)
        while(not result):
            vglClDilate(Rec_buffer, Rec_img_output , tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
            vglClMin(Rec_img_output, Rec_img_input, Rec_buffer)
            result = vglClEqual(Rec_buffer,Rec_img_output)
            

        #Kludge to stabilize the image
        #for n in range(40):
         #   vglClDilate(Rec_buffer, Rec_img_output , tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          #  vglClMin(Rec_img_output, Rec_img_input, Rec_buffer)
           # vglClDilate(Rec_buffer, Rec_img_output, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
            #vglClMin(Rec_img_output, Rec_img_input, Rec_buffer)

        #Runtime
        t0 = datetime.now()

        for i in range( nSteps ):
            while(not result):
              vglClDilate(Rec_buffer, Rec_img_output , tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
              vglClMin(Rec_img_output, Rec_img_input, Rec_buffer)
              result = vglClEqual(Rec_buffer,Rec_img_output)
 
        t1 = datetime.now()
        diff = t1 - t0
        med = (diff.total_seconds() * 1000) / nSteps
        msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo Reconstruct: " + str(med) + " ms\n"

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph.glyph_id, Rec_buffer)

    elif vGlyph.func == 'ShowImage':

        # Returns edge image based on glyph id
        ShowImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'image')

        if ShowImage_img_input is not None:

            # Rule3: In a sink glyph, images (one or more) can only be input parameters             
            vl.vglCheckContext(ShowImage_img_input,vl.VGL_RAM_CONTEXT())
            ShowImage_img_ndarray = VglImage.get_ipl(ShowImage_img_input)
            imshow(ShowImage_img_ndarray)

            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, None)

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
print("-------------------------------------------------------------")            
print(msg)
print("-------------------------------------------------------------")
