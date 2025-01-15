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

        elif vGlyph.func == 'vglClDilate': #Function Dilate
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")
        
            # Search the input image by connecting to the source glyph
            vglClDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

            # Search the output image by connecting to the source glyph
            vglClDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

            # Apply Dilate function
            vl.vglCheckContext(vglClDilate_img_output,vl.VGL_CL_CONTEXT())
            vglClDilate(vglClDilate_img_input, vglClDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))


            GlyphExecutedUpdate(vGlyph.glyph_id, vglClDilate_img_output)

        elif vGlyph.func == 'vglClErode': #Function Erode
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglClErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
            
            # Search the output image by connecting to the source glyph
            vglClErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        
            # Apply Erode function
            vl.vglCheckContext(vglClErode_img_output,vl.VGL_CL_CONTEXT())
            vglClErode(vglClErode_img_input, vglClErode_img_output, tratnum(vGlyph.lst_par[0].getValue()),np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
            

            GlyphExecutedUpdate(vGlyph.glyph_id, vglClErode_img_output)

            
        elif vGlyph.func == 'vglClConvolution': #Function Convolution
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglClConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
            
            # Search the output image by connecting to the source glyph
            vglClConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

            # Apply Convolution function
            #vl.vglCheckContext(vglClConvolution_img_output,vl.VGL_CL_CONTEXT())
            vglClConvolution(vglClConvolution_img_input, vglClConvolution_img_output,tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, vglClConvolution_img_output)

        elif vGlyph.func == 'vglClBlurSq3':  # Function blur
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglClBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

            # Search the output image by connecting to the source glyph
            vglClBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

            # Apply BlurSq3 function
            vglClBlurSq3(vglClBlurSq3_img_input, vglClBlurSq3_img_output)

            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, vglClBlurSq3_img_output)

        elif vGlyph.func == 'vglClRgb2Gray':  # Function Rgb2Gray
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglClRgb2Gray_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')

            # Search the output image by connecting to the source glyph
            vglClRgb2Gray_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

            # Apply Rgb2Gray function
            vglClRgb2Gray(vglClRgb2Gray_img_input, vglClRgb2Gray_img_output)

            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, vglClRgb2Gray_img_output)

        elif vGlyph.func == 'vglClSub': #Function Sub
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglClSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1')
            
            # Search the output image by connecting to the source glyph
            
            vglClSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

            vglClSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2')

            # Apply Sub Function       
            vglClSub(vglClSub_img_input1,vglClSub_img_input2,vglClSub_img_output)

            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, vglClSub_img_output)


        elif vGlyph.func == 'vglClThreshold': #Function Threshold
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")
        
            # Search the input image by connecting to the source glyph
            vglClThreshold_img_input = getImageInputByIdName(vGlyph.glyph_id, 'src')

            # Search the output image by connecting to the source glyph
            vglClThreshold_img_output = getImageInputByIdName(vGlyph.glyph_id, 'dst')

            # Apply Threshold function
            vglClThreshold(vglClThreshold_img_input, vglClThreshold_img_output, np.float32(vGlyph.lst_par[0].getValue()))


            GlyphExecutedUpdate(vGlyph.glyph_id, vglClThreshold_img_output)



        elif vGlyph.func == 'vglCl3dBlurSq3': #Function blur
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglCl3dBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
            
            # Search the output image by connecting to the source glyph
            vglCl3dBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

            # Apply BlurSq3 function
            vglCl3dBlurSq3(vglCl3dBlurSq3_img_input, vglCl3dBlurSq3_img_output)

            GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dBlurSq3_img_output)


        elif vGlyph.func == 'vglCl3dErode': #Function Erode
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglCl3dErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
            
            # Search the output image by connecting to the source glyph
            vglCl3dErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
        
            # Apply Erode function
            vl.vglCheckContext(vglCl3dErode_img_output,vl.VGL_CL_CONTEXT())
            vglCl3dErode(vglCl3dErode_img_input, vglCl3dErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), tratnum(vGlyph.lst_par[1].getValue()), tratnum(vGlyph.lst_par[2].getValue()),tratnum(vGlyph.lst_par[3].getValue()))
            
 
            GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dErode_img_output)


        elif vGlyph.func == 'vglCl3dConvolution': #Function Convolution
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglCl3dConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
            
            # Search the output image by connecting to the source glyph
            vglCl3dConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

            # Apply Convolution function
            #vl.vglCheckContext(vvglCl3dConvolution_img_output,vl.VGL_CL_CONTEXT())
            vglCl3dConvolution(vglCl3dConvolution_img_input, vglCl3dConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), tratnum(vGlyph.lst_par[1].getValue()), tratnum(vGlyph.lst_par[2].getValue()),tratnum(vGlyph.lst_par[3].getValue()))

 
            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dConvolution_img_output)

        elif vGlyph.func == 'vglCl3dDilate': #Function Dilate
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglCl3dDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
            
            # Search the output image by connecting to the source glyph
            vglCl3dDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')

            # Apply Dilate function
            vglCl3dDilate(vglCl3dDilate_img_input, vglCl3dDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), tratnum(vGlyph.lst_par[1].getValue()), tratnum(vGlyph.lst_par[2].getValue()),tratnum(vGlyph.lst_par[3].getValue()))


            GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dDilate_img_output)


        elif vGlyph.func == 'vglCl3dThreshold': #Function Threshold
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")
        
            # Search the input image by connecting to the source glyph
            vglCl3dThreshold_img_input = getImageInputByIdName(vGlyph.glyph_id, 'src')

            # Search the output image by connecting to the source glyph
            vglCl3dThreshold_img_output = getImageInputByIdName(vGlyph.glyph_id, 'dst')

            # Apply Threshold function
            vglCl3dThreshold(vglCl3dThreshold_img_input, vglCl3dThreshold_img_output, np.float32(vGlyph.lst_par[0].getValue()), np.float32(vGlyph.lst_par[1].getValue()))

            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dThreshold_img_output)




        elif vGlyph.func == 'vglClNdErode': #Function Erode
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglClNdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input')
            
            # Search the output image by connecting to the source glyph
            vglClNdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output')
            
            # Apply Erode function
            vl.vglCheckContext(vglClNdErode_img_output,vl.VGL_RAM_CONTEXT())

            window = getImageInputByIdName(vGlyph.glyph_id, 'window')

            vglClNdErode(vglClNdErode_img_input, vglClNdErode_img_output, window)
            
            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdErode_img_output)



execWorkflow(workspace)
