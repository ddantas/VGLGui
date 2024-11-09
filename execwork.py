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
vl.vglClInit(CPU) 

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
        
        #vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
        vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_2D_IMAGE(), vl.IMAGE_ND_ARRAY())
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

    elif vGlyph.func == 'vglShape': #Function Shape

        
        vglShape_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        
        #print(vglShape_img_input.prinfInfo())       
        vglShape = vl.VglShape()
        

              
        vglShape.constructorFromShapeNdimBps(vglShape.shape,int(vglShape_img_input.ndim))
        
        vglShape.shape[vl.VGL_SHAPE_NCHANNELS()] = 1
        vglShape.shape[vl.VGL_SHAPE_WIDTH()] = int(vGlyph.lst_par[0].getValue())
        vglShape.shape[vl.VGL_SHAPE_HEIGTH()] = int(vGlyph.lst_par[1].getValue())
        #vglShape.shape[vl.VGL_SHAPE_LENGTH()] = int(vGlyph.lst_par[3].getValue())
        vglShape.size = int(vGlyph.lst_par[0].getValue()) * int(vGlyph.lst_par[1].getValue())
        print(vglShape.printInfo())
        #print(vglShape.printInfo())
      
        GlyphExecutedUpdate(vGlyph.glyph_id, vglShape)


    elif vGlyph.func == 'vglStrel': #Function Erode
        print("-------------------------------------------------")
        print("A função " + vGlyph.func +" está sendo executada")
        print("-------------------------------------------------")
        
        vglShape = getImageInputByIdName(vGlyph.glyph_id, 'shape')

        ##CASO DO TYPE
        if (len(vGlyph.lst_par) == 2): 
          window = vl.VglStrEl()
          kernel_type_map = {
              'gaussian': 3,
              'cross': 2,
              'mean': 4,
              'cube': 1
          }
          input = vGlyph.lst_par[0].getValue().strip().lower()
          type = None
          for key in kernel_type_map.keys():
              if input.startswith(key):
                  type = kernel_type_map[key]
                  break          
          print(type)
          window.constructorFromTypeNdim(type, int(vGlyph.lst_par[1].getValue()))
          #print(window.getData())
          
        if(len(vGlyph.lst_par) == 1):
          str_list = vGlyph.lst_par[0].getValue()
          data = np.array(str_list, dtype=np.float32) 
          window = vl.VglStrEl()
          window.constructorFromDataVglShape(data,vglShape)
          #print(window.data)
          


        GlyphExecutedUpdate(vGlyph.glyph_id, window)

    elif vGlyph.func == 'vglClNdConvolution':

        vglClNdConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vl.vglCheckContext(vglClNdConvolution_img_input, vl.VGL_CL_CONTEXT());
        vglClNdConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vl.vglCheckContext(vglClNdConvolution_img_output, vl.VGL_CL_CONTEXT());
        window = getImageInputByIdName(vGlyph.glyph_id, 'window')
        vglClNdConvolution(vglClNdConvolution_img_input, vglClNdConvolution_img_output, window)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdConvolution_img_output)


    elif vGlyph.func == 'vglClNdCopy':

        vglClNdCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vl.vglCheckContext(vglClNdCopy_img_input, vl.VGL_CL_CONTEXT());
        vglClNdCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vl.vglCheckContext(vglClNdCopy_img_output, vl.VGL_CL_CONTEXT());
        vglClNdCopy(vglClNdCopy_img_input, vglClNdCopy_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdCopy_img_output)


    elif vGlyph.func == 'vglClNdDilate':

        vglClNdDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vl.vglCheckContext(vglClNdDilate_img_input, vl.VGL_CL_CONTEXT());
        vglClNdDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vl.vglCheckContext(vglClNdDilate_img_output, vl.VGL_CL_CONTEXT());
        window = getImageInputByIdName(vGlyph.glyph_id, 'window')
        vglClNdDilate(vglClNdDilate_img_input, vglClNdDilate_img_output, window)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdDilate_img_output)


    elif vGlyph.func == 'vglClNdErode':

        vglClNdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vl.vglCheckContext(vglClNdErode_img_input, vl.VGL_CL_CONTEXT());
        vglClNdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vl.vglCheckContext(vglClNdErode_img_output, vl.VGL_CL_CONTEXT());
        window = getImageInputByIdName(vGlyph.glyph_id, 'window')
        vglClNdErode(vglClNdErode_img_input, vglClNdErode_img_output, window)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdErode_img_output)


    elif vGlyph.func == 'vglClNdNot':

        vglClNdNot_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vl.vglCheckContext(vglClNdNot_img_input, vl.VGL_CL_CONTEXT());
        vglClNdNot_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vl.vglCheckContext(vglClNdNot_img_output, vl.VGL_CL_CONTEXT());
        vglClNdNot(vglClNdNot_img_input, vglClNdNot_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdNot_img_output)


    elif vGlyph.func == 'vglClNdThreshold':

        vglClNdThreshold_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
        vl.vglCheckContext(vglClNdThreshold_img_input, vl.VGL_CL_CONTEXT());
        vglClNdThreshold_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        vl.vglCheckContext(vglClNdThreshold_img_output, vl.VGL_CL_CONTEXT());
        vglClNdThreshold(vglClNdThreshold_img_input, vglClNdThreshold_img_output)

        GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdThreshold_img_output)

