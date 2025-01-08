#!/usr/bin/env python3

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
    print("Iniciando execWorkflow")
    if is_subworkflow:
        print(f"Executando no contexto de um sub-workflow. (Sub-workflow ID: {parent_workflow_id})")
    else:
        print("Executando no workflow principal.")
        
    if parent_workflow_id in processed_workflows:
        print(f"Workflow (ID: {parent_workflow_id}) já processado, evitando loop.")
        return
    
    # Adiciona o workflow ao conjunto de workflows processados
    # processed_workflows.add(parent_workflow_id)
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
        
        elif vGlyph.func == 'vglLoadImage':
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
            vGlyph.setGlyphReady(True)

            # Lê o sub-workflow
            sub_lstGlyph = []
            sub_lstConnection = []
            fileRead(workspace)
            print(f"Sub-workflow (ID: {vGlyph.glyph_id}) carregado")

            # Execução recursiva do sub-workflow
            execWorkflow(workspace)

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

execWorkflow(workspace)
