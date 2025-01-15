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
from readWorkflow import *  
import matplotlib.pyplot as mp


os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
sys.path.append(os.getcwd())


def imshow(im):
    plot = mp.imshow(im, cmap="gray", origin="upper", vmin=0, vmax=255)
    plot.set_interpolation('nearest')  # Configura a interpolação como "nearest"
    mp.colorbar()  # Adiciona uma barra de cores para facilitar a visualização dos valores
    mp.show()  # Exibe o gráfico

def tratnum(num):
    listnum = []
    for line in num:
        listnum.append(float(line))
    listnumpy = np.array(listnum, np.float32)
    return listnumpy

nSteps = 1
msg = ""
CPU = cl.device_type.CPU  # 2
GPU = cl.device_type.GPU  # 4
total = 0.0
vl.vglClInit(GPU)

processed_workflows = set()  # Usando um conjunto para armazenar IDs de workflows já processados

# Actions after glyph execution
def GlyphExecutedUpdate(GlyphExecutedUpdate_Glyph_Id, GlyphExecutedUpdate_image):
    # Rule10: Glyph becomes DONE = TRUE after its execution. Assign done to glyph
    setGlyphDoneId(GlyphExecutedUpdate_Glyph_Id)

    # Rule6: Edges whose source glyph has already been executed, and which therefore already had their image generated, have READY=TRUE (image ready to be processed).
    #        Reading the image from another glyph does not change this status. Check the list of connections
    setGlyphInputReadyByIdOut(GlyphExecutedUpdate_Glyph_Id)

    # Rule2: In a source glyph, images (one or more) can only be output parameters.
    setImageConnectionByOutputId(GlyphExecutedUpdate_Glyph_Id, GlyphExecutedUpdate_image)
    
workspace = Workspace()

fileRead(workspace)

def execWorkflow(workspace, is_subworkflow=False, parent_workflow_id=None):

    
    for vGlyph in workspace.lstGlyph:
        print(f"Processando o glyph: {vGlyph.glyph_id} com função {vGlyph.func}")

        print(vGlyph.getGlyphReady())
        try:
            if not vGlyph.getGlyphReady():
                break
        except ValueError:
            print("Rule9: Glyph not ready for processing: ", {vGlyph.glyph_id})


        if vGlyph.func == 'ProcedureEnd':
            print(f"Sub-workflow (ID: {parent_workflow_id}) finalizado, retornando ao workflow principal.")
            continue
        
        elif vGlyph.func == 'vglLoad2dImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")
            vglLoadImage_img_in_path = vGlyph.lst_par[0].getValue()
            vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())


            vl.vglLoadImage(vglLoadImage_img_input)
            if vglLoadImage_img_input.getVglShape().getNChannels() == 3:
                vl.rgb_to_rgba(vglLoadImage_img_input)

            vl.vglClUpload(vglLoadImage_img_input)
            GlyphExecutedUpdate(vGlyph.glyph_id, vglLoadImage_img_input)

        elif vGlyph.func == 'vglLoad3dImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")
            vglLoadImage_img_in_path = vGlyph.lst_par[0].getValue()
            vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_3D_IMAGE())

            vl.vglLoadImage(vglLoadImage_img_input)
            if vglLoadImage_img_input.getVglShape().getNChannels() == 3:
                vl.rgb_to_rgba(vglLoadImage_img_input)

            vl.vglClUpload(vglLoadImage_img_input)
            GlyphExecutedUpdate(vGlyph.glyph_id, vglLoadImage_img_input)


        elif vGlyph.func == 'vglLoadNdImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")
            vglLoadImage_img_in_path = vGlyph.lst_par[0].getValue()

            vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_2D_IMAGE(), vl.IMAGE_ND_ARRAY())

            vl.vglLoadImage(vglLoadImage_img_input)
            if vglLoadImage_img_input.getVglShape().getNChannels() == 3:
                vl.rgb_to_rgba(vglLoadImage_img_input)

            vl.vglClUpload(vglLoadImage_img_input)
            GlyphExecutedUpdate(vGlyph.glyph_id, vglLoadImage_img_input)

        elif vGlyph.func == 'vglCreateImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

            vglCreateImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img')
            vglCreateImage_RETVAL = vl.create_blank_image_as(vglCreateImage_img_input)
            vglCreateImage_RETVAL.set_oclPtr(vl.get_similar_oclPtr_object(vglCreateImage_img_input))
            vl.vglAddContext(vglCreateImage_RETVAL, vl.VGL_CL_CONTEXT())
            GlyphExecutedUpdate(vGlyph.glyph_id, vglCreateImage_RETVAL)

        elif vGlyph.func == 'ProcedureBegin':
            print(f"Iniciando sub-workflow (ID: {vGlyph.glyph_id})...")
            # Identifica a procedure e processa recursivamente
            sub_workspace = Workspace()  # Cria um novo workspace para o sub-workflow
            sub_workspace.lstGlyph = []  # Lista de glifos para o sub-workflow
            sub_workspace.lstConnection = []  # Lista de conexões para o sub-workflow
            is_subworkflow = True
            
            # Processa o sub-workflow recursivamente
            execWorkflow(sub_workspace, is_subworkflow=True, parent_workflow_id=vGlyph.glyph_id)
            continue  # Volta para o próximo glyph do workflow principal

        elif vGlyph.func == 'External Output (1)':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            o = getImageInputByIdName(vGlyph.glyph_id, 'o')
            GlyphExecutedUpdate(vGlyph.glyph_id, o)

        elif vGlyph.func == 'External Input (1)':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")
            
            o = getImageInputByIdName(vGlyph.glyph_id, 'i')
            GlyphExecutedUpdate(vGlyph.glyph_id, o)

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


            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id,Rec_img_output)

        elif vGlyph.func == 'vglShape': #Function Shape
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")
            
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

        elif vGlyph.func == 'vglSaveImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

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

        elif vGlyph.func == 'ShowImage':

            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")


            # Returns edge image based on glyph id
            ShowImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'image')

            if ShowImage_img_input is not None:
                # Rule3: In a sink glyph, images (one or more) can only be input parameters             
                vl.vglCheckContext(ShowImage_img_input, vl.VGL_RAM_CONTEXT())
                ShowImage_img_ndarray = VglImage.get_ipl(ShowImage_img_input)
                imshow(ShowImage_img_ndarray)

                # Actions after glyph execution
                GlyphExecutedUpdate(vGlyph.glyph_id, None)

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


        elif vGlyph.func == 'vglCl3dBlurSq3':

          vglCl3dBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dBlurSq3_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dBlurSq3_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dBlurSq3(vglCl3dBlurSq3_img_input, vglCl3dBlurSq3_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dBlurSq3_img_output)


        elif vGlyph.func == 'vglCl3dConvolution':

          vglCl3dConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dConvolution_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dConvolution(vglCl3dConvolution_img_input, vglCl3dConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dConvolution_img_output)


        elif vGlyph.func == 'vglCl3dCopy':

          vglCl3dCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dCopy_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dCopy_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dCopy(vglCl3dCopy_img_input, vglCl3dCopy_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dCopy_img_output)


        elif vGlyph.func == 'vglCl3dDilate':

          vglCl3dDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dDilate(vglCl3dDilate_img_input, vglCl3dDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dDilate_img_output)


        elif vGlyph.func == 'vglCl3dErode':

          vglCl3dErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dErode(vglCl3dErode_img_input, vglCl3dErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dErode_img_output)


        elif vGlyph.func == 'vglCl3dMax':

          vglCl3dMax_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
          vl.vglCheckContext(vglCl3dMax_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dMax_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
          vl.vglCheckContext(vglCl3dMax_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dMax_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dMax_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dMax(vglCl3dMax_img_input1, vglCl3dMax_img_input2, vglCl3dMax_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dMax_img_output)


        elif vGlyph.func == 'vglCl3dMin':

          vglCl3dMin_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
          vl.vglCheckContext(vglCl3dMin_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dMin_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
          vl.vglCheckContext(vglCl3dMin_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dMin_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dMin_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dMin(vglCl3dMin_img_input1, vglCl3dMin_img_input2, vglCl3dMin_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dMin_img_output)


        elif vGlyph.func == 'vglCl3dNot':

          vglCl3dNot_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dNot_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dNot_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dNot_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dNot(vglCl3dNot_img_input, vglCl3dNot_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dNot_img_output)


        elif vGlyph.func == 'vglCl3dSub':

          vglCl3dSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
          vl.vglCheckContext(vglCl3dSub_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
          vl.vglCheckContext(vglCl3dSub_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dSub_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dSub(vglCl3dSub_img_input1, vglCl3dSub_img_input2, vglCl3dSub_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dSub_img_output)


        elif vGlyph.func == 'vglCl3dSum':

          vglCl3dSum_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
          vl.vglCheckContext(vglCl3dSum_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dSum_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
          vl.vglCheckContext(vglCl3dSum_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dSum_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dSum_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dSum(vglCl3dSum_img_input1, vglCl3dSum_img_input2, vglCl3dSum_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dSum_img_output)


        elif vGlyph.func == 'vglCl3dThreshold':

          vglCl3dThreshold_src = getImageInputByIdName(vGlyph.glyph_id, 'src')
          vl.vglCheckContext(vglCl3dThreshold_src, vl.VGL_CL_CONTEXT());
          vglCl3dThreshold_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst')
          vl.vglCheckContext(vglCl3dThreshold_dst, vl.VGL_CL_CONTEXT());
          vglCl3dThreshold(vglCl3dThreshold_src, vglCl3dThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dThreshold_dst)


        elif vGlyph.func == 'vglClBlurSq3':

          vglClBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClBlurSq3_img_input, vl.VGL_CL_CONTEXT());
          vglClBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClBlurSq3_img_output, vl.VGL_CL_CONTEXT());
          vglClBlurSq3(vglClBlurSq3_img_input, vglClBlurSq3_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClBlurSq3_img_output)


        elif vGlyph.func == 'vglClConvolution':

          vglClConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglClConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClConvolution_img_output, vl.VGL_CL_CONTEXT());
          vglClConvolution(vglClConvolution_img_input, vglClConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClConvolution_img_output)


        elif vGlyph.func == 'vglClCopy':

          vglClCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClCopy_img_input, vl.VGL_CL_CONTEXT());
          vglClCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClCopy_img_output, vl.VGL_CL_CONTEXT());
          vglClCopy(vglClCopy_img_input, vglClCopy_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClCopy_img_output)


        elif vGlyph.func == 'vglClDilate':

          vglClDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClDilate(vglClDilate_img_input, vglClDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClDilate_img_output)


        elif vGlyph.func == 'vglClErode':

          vglClErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClErode_img_input, vl.VGL_CL_CONTEXT());
          vglClErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClErode_img_output, vl.VGL_CL_CONTEXT());
          vglClErode(vglClErode_img_input, vglClErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClErode_img_output)


        elif vGlyph.func == 'vglClInvert':

          vglClInvert_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClInvert_img_input, vl.VGL_CL_CONTEXT());
          vglClInvert_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClInvert_img_output, vl.VGL_CL_CONTEXT());
          vglClInvert(vglClInvert_img_input, vglClInvert_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClInvert_img_output)


        elif vGlyph.func == 'vglClMax':

          vglClMax_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
          vl.vglCheckContext(vglClMax_img_input1, vl.VGL_CL_CONTEXT());
          vglClMax_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
          vl.vglCheckContext(vglClMax_img_input2, vl.VGL_CL_CONTEXT());
          vglClMax_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClMax_img_output, vl.VGL_CL_CONTEXT());
          vglClMax(vglClMax_img_input1, vglClMax_img_input2, vglClMax_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClMax_img_output)


        elif vGlyph.func == 'vglClMin':

          vglClMin_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
          vl.vglCheckContext(vglClMin_img_input1, vl.VGL_CL_CONTEXT());
          vglClMin_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
          vl.vglCheckContext(vglClMin_img_input2, vl.VGL_CL_CONTEXT());
          vglClMin_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClMin_img_output, vl.VGL_CL_CONTEXT());
          vglClMin(vglClMin_img_input1, vglClMin_img_input2, vglClMin_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClMin_img_output)


        elif vGlyph.func == 'vglClRgb2Gray':

          vglClRgb2Gray_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClRgb2Gray_img_input, vl.VGL_CL_CONTEXT());
          vglClRgb2Gray_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClRgb2Gray_img_output, vl.VGL_CL_CONTEXT());
          vglClRgb2Gray(vglClRgb2Gray_img_input, vglClRgb2Gray_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClRgb2Gray_img_output)


        elif vGlyph.func == 'vglClSub':

          vglClSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
          vl.vglCheckContext(vglClSub_img_input1, vl.VGL_CL_CONTEXT());
          vglClSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
          vl.vglCheckContext(vglClSub_img_input2, vl.VGL_CL_CONTEXT());
          vglClSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClSub_img_output, vl.VGL_CL_CONTEXT());
          vglClSub(vglClSub_img_input1, vglClSub_img_input2, vglClSub_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSub_img_output)


        elif vGlyph.func == 'vglClSum':

          vglClSum_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
          vl.vglCheckContext(vglClSum_img_input1, vl.VGL_CL_CONTEXT());
          vglClSum_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')
          vl.vglCheckContext(vglClSum_img_input2, vl.VGL_CL_CONTEXT());
          vglClSum_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClSum_img_output, vl.VGL_CL_CONTEXT());
          vglClSum(vglClSum_img_input1, vglClSum_img_input2, vglClSum_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSum_img_output)


        elif vGlyph.func == 'vglClSwapRgb':

          vglClSwapRgb_src = getImageInputByIdName(vGlyph.glyph_id, 'src')
          vl.vglCheckContext(vglClSwapRgb_src, vl.VGL_CL_CONTEXT());
          vglClSwapRgb_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst')
          vl.vglCheckContext(vglClSwapRgb_dst, vl.VGL_CL_CONTEXT());
          vglClSwapRgb(vglClSwapRgb_src, vglClSwapRgb_dst)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSwapRgb_dst)


        elif vGlyph.func == 'vglClThreshold':

          vglClThreshold_src = getImageInputByIdName(vGlyph.glyph_id, 'src')
          vl.vglCheckContext(vglClThreshold_src, vl.VGL_CL_CONTEXT());
          vglClThreshold_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst')
          vl.vglCheckContext(vglClThreshold_dst, vl.VGL_CL_CONTEXT());
          vglClThreshold(vglClThreshold_src, vglClThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClThreshold_dst)


        elif vGlyph.func == 'vglCl3dFuzzyAlgDilate':

          vglCl3dFuzzyAlgDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyAlgDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyAlgDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgDilate(vglCl3dFuzzyAlgDilate_img_input, vglCl3dFuzzyAlgDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyAlgDilate_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyAlgErode':

          vglCl3dFuzzyAlgErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyAlgErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyAlgErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgErode(vglCl3dFuzzyAlgErode_img_input, vglCl3dFuzzyAlgErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyAlgErode_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyArithDilate':

          vglCl3dFuzzyArithDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyArithDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyArithDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithDilate(vglCl3dFuzzyArithDilate_img_input, vglCl3dFuzzyArithDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyArithDilate_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyArithErode':

          vglCl3dFuzzyArithErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyArithErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyArithErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithErode(vglCl3dFuzzyArithErode_img_input, vglCl3dFuzzyArithErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyArithErode_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyBoundDilate':

          vglCl3dFuzzyBoundDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyBoundDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyBoundDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundDilate(vglCl3dFuzzyBoundDilate_img_input, vglCl3dFuzzyBoundDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyBoundDilate_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyBoundErode':

          vglCl3dFuzzyBoundErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyBoundErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyBoundErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundErode(vglCl3dFuzzyBoundErode_img_input, vglCl3dFuzzyBoundErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyBoundErode_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyDaPDilate':

          vglCl3dFuzzyDaPDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyDaPDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyDaPDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPDilate(vglCl3dFuzzyDaPDilate_img_input, vglCl3dFuzzyDaPDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDaPDilate_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyDaPErode':

          vglCl3dFuzzyDaPErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyDaPErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyDaPErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPErode(vglCl3dFuzzyDaPErode_img_input, vglCl3dFuzzyDaPErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDaPErode_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyDrasticDilate':

          vglCl3dFuzzyDrasticDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyDrasticDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyDrasticDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticDilate(vglCl3dFuzzyDrasticDilate_img_input, vglCl3dFuzzyDrasticDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDrasticDilate_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyDrasticErode':

          vglCl3dFuzzyDrasticErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyDrasticErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyDrasticErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticErode(vglCl3dFuzzyDrasticErode_img_input, vglCl3dFuzzyDrasticErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDrasticErode_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyGeoDilate':

          vglCl3dFuzzyGeoDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyGeoDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyGeoDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoDilate(vglCl3dFuzzyGeoDilate_img_input, vglCl3dFuzzyGeoDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyGeoDilate_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyGeoErode':

          vglCl3dFuzzyGeoErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyGeoErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyGeoErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoErode(vglCl3dFuzzyGeoErode_img_input, vglCl3dFuzzyGeoErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyGeoErode_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyHamacherDilate':

          vglCl3dFuzzyHamacherDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyHamacherDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyHamacherDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherDilate(vglCl3dFuzzyHamacherDilate_img_input, vglCl3dFuzzyHamacherDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyHamacherDilate_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyHamacherErode':

          vglCl3dFuzzyHamacherErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyHamacherErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyHamacherErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherErode(vglCl3dFuzzyHamacherErode_img_input, vglCl3dFuzzyHamacherErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyHamacherErode_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyStdDilate':

          vglCl3dFuzzyStdDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyStdDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyStdDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdDilate(vglCl3dFuzzyStdDilate_img_input, vglCl3dFuzzyStdDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyStdDilate_img_output)


        elif vGlyph.func == 'vglCl3dFuzzyStdErode':

          vglCl3dFuzzyStdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglCl3dFuzzyStdErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglCl3dFuzzyStdErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdErode(vglCl3dFuzzyStdErode_img_input, vglCl3dFuzzyStdErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyStdErode_img_output)


        elif vGlyph.func == 'vglClFuzzyAlgDilate':

          vglClFuzzyAlgDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyAlgDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyAlgDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgDilate(vglClFuzzyAlgDilate_img_input, vglClFuzzyAlgDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyAlgDilate_img_output)


        elif vGlyph.func == 'vglClFuzzyAlgErode':

          vglClFuzzyAlgErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyAlgErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyAlgErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgErode(vglClFuzzyAlgErode_img_input, vglClFuzzyAlgErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyAlgErode_img_output)


        elif vGlyph.func == 'vglClFuzzyArithDilate':

          vglClFuzzyArithDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyArithDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyArithDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithDilate(vglClFuzzyArithDilate_img_input, vglClFuzzyArithDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyArithDilate_img_output)


        elif vGlyph.func == 'vglClFuzzyArithErode':

          vglClFuzzyArithErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyArithErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyArithErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithErode(vglClFuzzyArithErode_img_input, vglClFuzzyArithErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyArithErode_img_output)


        elif vGlyph.func == 'vglClFuzzyBoundDilate':

          vglClFuzzyBoundDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyBoundDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyBoundDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundDilate(vglClFuzzyBoundDilate_img_input, vglClFuzzyBoundDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyBoundDilate_img_output)


        elif vGlyph.func == 'vglClFuzzyBoundErode':

          vglClFuzzyBoundErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyBoundErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyBoundErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundErode(vglClFuzzyBoundErode_img_input, vglClFuzzyBoundErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyBoundErode_img_output)


        elif vGlyph.func == 'vglClFuzzyDaPDilate':

          vglClFuzzyDaPDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyDaPDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyDaPDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPDilate(vglClFuzzyDaPDilate_img_input, vglClFuzzyDaPDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDaPDilate_img_output)


        elif vGlyph.func == 'vglClFuzzyDaPErode':

          vglClFuzzyDaPErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyDaPErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyDaPErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPErode(vglClFuzzyDaPErode_img_input, vglClFuzzyDaPErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDaPErode_img_output)


        elif vGlyph.func == 'vglClFuzzyDrasticDilate':

          vglClFuzzyDrasticDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyDrasticDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyDrasticDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticDilate(vglClFuzzyDrasticDilate_img_input, vglClFuzzyDrasticDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDrasticDilate_img_output)


        elif vGlyph.func == 'vglClFuzzyDrasticErode':

          vglClFuzzyDrasticErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyDrasticErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyDrasticErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticErode(vglClFuzzyDrasticErode_img_input, vglClFuzzyDrasticErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDrasticErode_img_output)


        elif vGlyph.func == 'vglClFuzzyGeoDilate':

          vglClFuzzyGeoDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyGeoDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyGeoDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoDilate(vglClFuzzyGeoDilate_img_input, vglClFuzzyGeoDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyGeoDilate_img_output)


        elif vGlyph.func == 'vglClFuzzyGeoErode':

          vglClFuzzyGeoErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyGeoErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyGeoErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoErode(vglClFuzzyGeoErode_img_input, vglClFuzzyGeoErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyGeoErode_img_output)


        elif vGlyph.func == 'vglClFuzzyHamacherDilate':

          vglClFuzzyHamacherDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyHamacherDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyHamacherDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherDilate(vglClFuzzyHamacherDilate_img_input, vglClFuzzyHamacherDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyHamacherDilate_img_output)


        elif vGlyph.func == 'vglClFuzzyHamacherErode':

          vglClFuzzyHamacherErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyHamacherErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyHamacherErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherErode(vglClFuzzyHamacherErode_img_input, vglClFuzzyHamacherErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyHamacherErode_img_output)


        elif vGlyph.func == 'vglClFuzzyStdDilate':

          vglClFuzzyStdDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyStdDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyStdDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdDilate(vglClFuzzyStdDilate_img_input, vglClFuzzyStdDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyStdDilate_img_output)


        elif vGlyph.func == 'vglClFuzzyStdErode':

          vglClFuzzyStdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
          vl.vglCheckContext(vglClFuzzyStdErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
          vl.vglCheckContext(vglClFuzzyStdErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdErode(vglClFuzzyStdErode_img_input, vglClFuzzyStdErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyStdErode_img_output)


execWorkflow(workspace)
