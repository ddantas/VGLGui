#!/usr/bin/env python3
from vgl_lib.vglClUtil import vglClEqual

from vgl_lib.vglImage import VglImage
import pyopencl as cl
import vgl_lib as vl
import numpy as np
from cl2py_shaders import * 
from cl2py_ND import *
import os
import sys
from readWorkflow import *
import time as t
from datetime import datetime
from readWorkflow import *
import matplotlib.pyplot as mp


os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
sys.path.append(os.getcwd())


def imshow(im):
    plot = mp.imshow(im, cmap="gray", origin="upper", vmin=0, vmax=255)
    plot.set_interpolation("nearest")
    mp.show() 


def tratnum(num):
    listnum = []
    for line in num:
        listnum.append(float(line))
    listnumpy = np.array(listnum, np.float32)
    return listnumpy


nSteps = 1
msg = ""
CPU = cl.device_type.CPU 
GPU = cl.device_type.GPU
total = 0.0
vl.vglClInit(GPU)


workspace = Workspace()  # Inicializa seu workspace principal
fileRead(workspace)  # Lê os dados no workspace

def GlyphExecutedUpdate(GlyphExecutedUpdate_Glyph_Id, GlyphExecutedUpdate_image, workspace):
    # Rule10: Glyph becomes DONE = TRUE after its execution. Assign done to glyph
    setGlyphDoneId(GlyphExecutedUpdate_Glyph_Id,workspace)

    # Rule6: Edges whose source glyph has already been executed, and which therefore already had their image generated, have READY=TRUE (image ready to be processed).
    #        Reading the image from another glyph does not change this status. Check the list of connections
    setGlyphInputReadyByIdOut(GlyphExecutedUpdate_Glyph_Id, workspace)

    # Rule2: In a source glyph, images (one or more) can only be output parameters.
    setImageConnectionByOutputId(GlyphExecutedUpdate_Glyph_Id, GlyphExecutedUpdate_image, workspace)
    

def execute_workspace(workspace):
    print(f"Processando workspace: {workspace}")

    for vGlyph in workspace.lstGlyph:
        if vGlyph.func == 'vglLoad2dImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")
            vglLoadImage_img_in_path = vGlyph.lst_par[0].getValue()
            vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())


            vl.vglLoadImage(vglLoadImage_img_input)
            if vglLoadImage_img_input.getVglShape().getNChannels() == 3:
                vl.rgb_to_rgba(vglLoadImage_img_input)

            vl.vglClUpload(vglLoadImage_img_input)
            GlyphExecutedUpdate(vGlyph.glyph_id, vglLoadImage_img_input, workspace)



        elif vGlyph.func == 'vglCreateImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")
            print("Glyph ID:", vGlyph.glyph_id)

            vglCreateImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img', workspace)
            
            if vglCreateImage_img_input is None:
                print(f"Nenhuma imagem encontrada para glyph_id={vGlyph.glyph_id} e name=img")
                # Aqui você pode decidir o que fazer: talvez criar uma imagem padrão ou continuar sem imagem
                return  # Ou talvez você queira continuar com um comportamento alternativo
            else:
                vglCreateImage_RETVAL = vl.create_blank_image_as(vglCreateImage_img_input)
                vglCreateImage_RETVAL.set_oclPtr(vl.get_similar_oclPtr_object(vglCreateImage_img_input))
                vl.vglAddContext(vglCreateImage_RETVAL, vl.VGL_CL_CONTEXT())
                GlyphExecutedUpdate(vGlyph.glyph_id, vglCreateImage_RETVAL, workspace)


        elif vGlyph.func == "ShowImage":

            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

            print("Glyph ID:", vGlyph.glyph_id)
            # Returns edge image based on glyph id
            ShowImage_img_input = getImageInputByIdName(
                vGlyph.glyph_id, "image", workspace
            )

            if ShowImage_img_input is not None:
                # Rule3: In a sink glyph, images (one or more) can only be input parameters
                vl.vglCheckContext(ShowImage_img_input, vl.VGL_RAM_CONTEXT())
                ShowImage_img_ndarray = VglImage.get_ipl(ShowImage_img_input)
                imshow(ShowImage_img_ndarray)

                # Actions after glyph execution
                GlyphExecutedUpdate(vGlyph.glyph_id, None, workspace)


        elif vGlyph.func == 'External Input (1)':
            print("-------------------------------------------------")
            print(f"A função {vGlyph.func} está sendo executada")
            print("-------------------------------------------------")
            
            glyph = next((g for g in workspace.lstGlyph if g.glyph_id == vGlyph.glyph_id), None)
            if glyph:
                print(f"Glyph com ID {glyph.glyph_id} encontrado.")
                o = getImageInputByIdName(glyph.glyph_id, 'i', workspace)

                if hasattr(workspace, "subWorkspaces") and workspace.subWorkspaces:
                    for subWorkspace in workspace.subWorkspaces:
                        print(f"Enviando dados para o subworkspace {subWorkspace}")
                        execute_workspace(subWorkspace)  # Chamada recursiva para processar o subworkspace
                
                GlyphExecutedUpdate(glyph.glyph_id, o, workspace)

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
            GlyphExecutedUpdate(vGlyph.glyph_id, vglLoadImage_img_input, workspace)

        elif vGlyph.func == 'vglShape': #Function Shape
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")
            
            vglShape_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
            
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
        
            GlyphExecutedUpdate(vGlyph.glyph_id, vglShape, workspace)

        elif vGlyph.func == 'vglStrel': #Function Erode
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")
            
            vglShape = getImageInputByIdName(vGlyph.glyph_id, 'shape', workspace)

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
            


            GlyphExecutedUpdate(vGlyph.glyph_id, window, workspace)


        elif vGlyph.func == 'vglClNdErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func +" está sendo executada")
          print("-------------------------------------------------")

          vglClNdConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClNdConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglClNdConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClNdConvolution_img_output, vl.VGL_CL_CONTEXT());
          window = getImageInputByIdName(vGlyph.glyph_id, 'window', workspace)
          vglClNdErode(vglClNdConvolution_img_input, vglClNdConvolution_img_output, window)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdConvolution_img_output, workspace)

        elif vGlyph.func == 'vglSaveImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

            # Returns edge image based on glyph id
            vglSaveImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'image', workspace)
            print("path = ",vGlyph.lst_par[0].getValue())
            print("vglSaveImage_img_input = ",vglSaveImage_img_input)
            if vglSaveImage_img_input is not None:

              # SAVING IMAGE img
              vpath = vGlyph.lst_par[0].getValue()

              # Rule3: In a sink glyph, images (one or more) can only be input parameters
              vl.vglCheckContext(vglSaveImage_img_input,vl.VGL_RAM_CONTEXT())             
              vl.vglSaveImage(vpath, vglSaveImage_img_input)
              

              # Actions after glyph execution
              GlyphExecutedUpdate(vGlyph.glyph_id, None, workspace)


        elif vGlyph.func == 'vglClRgb2Gray':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func +" está sendo executada")
          print("-------------------------------------------------")
          print('s')
          
        #   vglClRgb2Gray_img_input = getImageInputByIdName(vGlyph.glyph_id, 'i', workspace)
          vglClRgb2Gray_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClRgb2Gray_img_input, vl.VGL_CL_CONTEXT());
          vglClRgb2Gray_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClRgb2Gray_img_output, vl.VGL_CL_CONTEXT());
          vglClRgb2Gray(vglClRgb2Gray_img_input, vglClRgb2Gray_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClRgb2Gray_img_output, workspace)

        elif vGlyph.func == 'vglClErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func +" está sendo executada")
          print("-------------------------------------------------")
          vglClErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          print(f"Imagem após vglClErode: {vglClErode_img_input.shape}")


          vl.vglCheckContext(vglClErode_img_input, vl.VGL_CL_CONTEXT());
          vglClErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClErode_img_output, vl.VGL_CL_CONTEXT());
          vglClErode(vglClErode_img_input, vglClErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          
          print(f"Imagem após vglClErode: {vglClErode_img_output.shape}")
          GlyphExecutedUpdate(vGlyph.glyph_id, vglClErode_img_output, workspace)

        elif vGlyph.func == 'vglClConvolution':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func +" está sendo executada")
          print("-------------------------------------------------")

          vglClConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglClConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClConvolution_img_output, vl.VGL_CL_CONTEXT());
          vglClConvolution(vglClConvolution_img_input, vglClConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClConvolution_img_output, workspace)

        elif vGlyph.func == 'vglClDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func +" está sendo executada")
          print("-------------------------------------------------")

          vglClDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClDilate(vglClDilate_img_input, vglClDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClDilate_img_output, workspace)
          
        elif vGlyph.func == 'vglClSub':

          vglClSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1', workspace)
          vl.vglCheckContext(vglClSub_img_input1, vl.VGL_CL_CONTEXT());
          vglClSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2', workspace)
          vl.vglCheckContext(vglClSub_img_input2, vl.VGL_CL_CONTEXT());
          vglClSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClSub_img_output, vl.VGL_CL_CONTEXT());
          vglClSub(vglClSub_img_input1, vglClSub_img_input2, vglClSub_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSub_img_output, workspace)

        elif vGlyph.func == 'vglClThreshold':

          vglClThreshold_src = getImageInputByIdName(vGlyph.glyph_id, 'src', workspace)
          vl.vglCheckContext(vglClThreshold_src, vl.VGL_CL_CONTEXT());
          vglClThreshold_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst', workspace)
          vl.vglCheckContext(vglClThreshold_dst, vl.VGL_CL_CONTEXT());
          vglClThreshold(vglClThreshold_src, vglClThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClThreshold_dst, workspace)
        
        elif vGlyph.func == 'Reconstruct': #Function Reconstruct
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")
        
            # Search the input image by connecting to the source glyph
            Rec_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)

            

            # Search the output image by connecting to the source glyph
            Rec_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)

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
            GlyphExecutedUpdate(vGlyph.glyph_id,Rec_img_output, workspace)


    # Se houver sub-workspaces, realiza a chamada recursiva para processá-los também
    if hasattr(workspace, "subWorkspaces") and workspace.subWorkspaces:
        for subWorkspace in workspace.subWorkspaces:
            print(f"Entrando no subWorkspace: {subWorkspace}")
            # Chamada recursiva para o sub-workspace
            execute_workspace(subWorkspace)


# Executa os glifos no workspace principal
execute_workspace(workspace)
