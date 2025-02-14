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


nSteps = 100
msg = ""
CPU = cl.device_type.CPU 
GPU = cl.device_type.GPU
total = 0.0
vl.vglClInit(GPU)


workspace = Workspace()
fileRead(workspace)

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
    msg = ""
    total = 0.0
    for vGlyph in workspace.lstGlyph:
        try:
            # Verifica se o glifo está pronto para ser processado
            if vGlyph.getGlyphReady() is False:
                print(f"ERROR: Glyph {vGlyph.glyph_id} not ready for processing.")
                break
        except Exception as e:
            print(f"Unexpected error while processing glyph {vGlyph.glyph_id}: {e}")
            
        if vGlyph.func == 'vglLoad2dImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")
            vglLoadImage_img_in_path = vGlyph.lst_par[0].getValue()
            vglLoadImage_img_input = vl.VglImage(vglLoadImage_img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())

            # print(vGlyph.getStatus)
            vl.vglLoadImage(vglLoadImage_img_input)
            if vglLoadImage_img_input.getVglShape().getNChannels() == 3:
                vl.rgb_to_rgba(vglLoadImage_img_input)

            vl.vglClUpload(vglLoadImage_img_input)
            GlyphExecutedUpdate(vGlyph.glyph_id, vglLoadImage_img_input, workspace)

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
            GlyphExecutedUpdate(vGlyph.glyph_id, vglLoadImage_img_input, workspace)

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

        elif vGlyph.func == 'vglCreateImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")
            print("Glyph ID:", vGlyph.glyph_id)

            vglCreateImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img', workspace)
            
            if vglCreateImage_img_input is None:
                print(f"Nenhuma imagem encontrada para glyph_id={vGlyph.glyph_id} e name=img")
                return
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
            ShowImage_img_input = getImageInputByIdName(vGlyph.glyph_id, "image", workspace)

            if ShowImage_img_input is not None:
                # Verifica se a imagem foi movida para o workspace principal
                vl.vglCheckContext(ShowImage_img_input, vl.VGL_RAM_CONTEXT())
                ShowImage_img_ndarray = VglImage.get_ipl(ShowImage_img_input)
                imshow(ShowImage_img_ndarray)

                # Após exibir a imagem, atualiza o estado do Glyph
                GlyphExecutedUpdate(vGlyph.glyph_id, None, workspace)
            else:
                print(f"Aviso: Nenhuma imagem encontrada no workspace principal para o Glyph ID {vGlyph.glyph_id}")


        elif vGlyph.func == 'External Input (1)':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

            # Enviar dados para o subworkspace
            input_value_sub = getImageInputByIdName(vGlyph.glyph_id, 'i', workspace)
            if hasattr(workspace, "subWorkspaces") and workspace.subWorkspaces:
                for subWorkspace in workspace.subWorkspaces:
                    GlyphExecutedUpdate(vGlyph.glyph_id, input_value_sub, subWorkspace)



        elif vGlyph.func == 'External Output (1)':
            print("-------------------------------------------------")
            print(f"A função {vGlyph.func} está sendo executada")
            print("-------------------------------------------------")
            
            # Buscando o glifo com o ID correspondente
            glyph = next((g for g in workspace.lstGlyph if g.glyph_id == vGlyph.glyph_id), None)
            if glyph:
                print(f"Glyph com ID {glyph.glyph_id} encontrado.")
                
                # Procurando imagem de saída associada ao glifo
                o = getImageInputByIdName(glyph.glyph_id, 'o', workspace)
                if o is None:
                    print(f"Imagem para glyph_id={glyph.glyph_id} e name='o' não encontrada.")
                else:
                    print(f"Imagem encontrada: {o}")

                # Se a imagem não for None, envie os dados para a procedure
                if o is not None:
                    print(f"Enviando dados para a procedure.")
                    
                    # Envia os dados para a procedure (sub-workspace)
                    GlyphExecutedUpdate(glyph.glyph_id, o, workspace)  # Envia para a procedure
                    print(f"Dados enviados para a procedure.")
                else:
                    print(f"Nenhuma imagem para enviar à procedure.")
            else:
                print(f"Glifo com ID {vGlyph.glyph_id} não encontrado.")

        elif vGlyph.func == 'ProcedureBegin':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")
            
            # Processa a Procedure chamando os subWorkspaces
            if hasattr(workspace, "subWorkspaces") and workspace.subWorkspaces:
                for subWorkspace in workspace.subWorkspaces:
                    execute_workspace(subWorkspace)  # Executa a procedure

                    # Obtém os dados de saída da procedure
                    o = getImageInputByIdName(vGlyph.glyph_id, 'o', subWorkspace)
                    
                    # Envia os dados para o workspace principal
                    if o is not None:
                        print(f"Enviando dados da procedure para o workspace principal.")
                        GlyphExecutedUpdate(vGlyph.glyph_id, o, workspace)  # Envia para o workspace principal
                        print(f"Dados enviados e workspace principal atualizado.")
                    else:
                        print(f"Nenhuma imagem para enviar ao workspace principal.")
            
            print("-------------------------------------------------")
            print("Retornando ao workspace principal após ProcedureBegin")

        elif vGlyph.func == 'vglSaveImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

            # Returns edge image based on glyph id
            vglSaveImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'image', workspace)

            if vglSaveImage_img_input is not None:

                # SAVING IMAGE img
                vpath = vGlyph.lst_par[0].getValue()

                # Rule3: In a sink glyph, images (one or more) can only be input parameters
                vl.vglCheckContext(vglSaveImage_img_input,vl.VGL_RAM_CONTEXT())             
                vl.vglSaveImage(vpath, vglSaveImage_img_input)

                # Actions after glyph execution
                GlyphExecutedUpdate(vGlyph.glyph_id, None, workspace)

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
            t1 = datetime.now()
            diff = t1 - t0
            media = round((diff.total_seconds() * 1000) / nSteps, 3)
            msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do método Reconstruct: " + str(media) + " ms\n"
            total = total + media
            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id,Rec_img_output, workspace)

        elif vGlyph.func == 'vglClNdConvolution':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClNdConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClNdConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglClNdConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClNdConvolution_img_output, vl.VGL_CL_CONTEXT());
          window = getImageInputByIdName(vGlyph.glyph_id, 'window', workspace)
          vglClNdConvolution(vglClNdConvolution_img_input, vglClNdConvolution_img_output, window)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClNdConvolution(vglClNdConvolution_img_input, vglClNdConvolution_img_output, window)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClNdConvolution: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdConvolution_img_output, workspace)


        elif vGlyph.func == 'vglClNdCopy':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClNdCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClNdCopy_img_input, vl.VGL_CL_CONTEXT());
          vglClNdCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClNdCopy_img_output, vl.VGL_CL_CONTEXT());
          vglClNdCopy(vglClNdCopy_img_input, vglClNdCopy_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClNdCopy(vglClNdCopy_img_input, vglClNdCopy_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClNdCopy: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdCopy_img_output, workspace)


        elif vGlyph.func == 'vglClNdDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClNdDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClNdDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClNdDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClNdDilate_img_output, vl.VGL_CL_CONTEXT());
          window = getImageInputByIdName(vGlyph.glyph_id, 'window', workspace)
          vglClNdDilate(vglClNdDilate_img_input, vglClNdDilate_img_output, window)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClNdDilate(vglClNdDilate_img_input, vglClNdDilate_img_output, window)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClNdDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdDilate_img_output, workspace)


        elif vGlyph.func == 'vglClNdErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClNdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClNdErode_img_input, vl.VGL_CL_CONTEXT());
          vglClNdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClNdErode_img_output, vl.VGL_CL_CONTEXT());
          window = getImageInputByIdName(vGlyph.glyph_id, 'window', workspace)
          vglClNdErode(vglClNdErode_img_input, vglClNdErode_img_output, window)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClNdErode(vglClNdErode_img_input, vglClNdErode_img_output, window)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClNdErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdErode_img_output, workspace)


        elif vGlyph.func == 'vglClNdNot':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClNdNot_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClNdNot_img_input, vl.VGL_CL_CONTEXT());
          vglClNdNot_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClNdNot_img_output, vl.VGL_CL_CONTEXT());
          vglClNdNot(vglClNdNot_img_input, vglClNdNot_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClNdNot(vglClNdNot_img_input, vglClNdNot_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClNdNot: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdNot_img_output, workspace)


        elif vGlyph.func == 'vglClNdThreshold':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClNdThreshold_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClNdThreshold_img_input, vl.VGL_CL_CONTEXT());
          vglClNdThreshold_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClNdThreshold_img_output, vl.VGL_CL_CONTEXT());
          vglClNdThreshold(vglClNdThreshold_img_input, vglClNdThreshold_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClNdThreshold(vglClNdThreshold_img_input, vglClNdThreshold_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClNdThreshold: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdThreshold_img_output, workspace)


        elif vGlyph.func == 'vglCl3dBlurSq3':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dBlurSq3_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dBlurSq3_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dBlurSq3(vglCl3dBlurSq3_img_input, vglCl3dBlurSq3_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dBlurSq3(vglCl3dBlurSq3_img_input, vglCl3dBlurSq3_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dBlurSq3: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dBlurSq3_img_output, workspace)


        elif vGlyph.func == 'vglCl3dConvolution':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dConvolution_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dConvolution(vglCl3dConvolution_img_input, vglCl3dConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dConvolution(vglCl3dConvolution_img_input, vglCl3dConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dConvolution: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dConvolution_img_output, workspace)


        elif vGlyph.func == 'vglCl3dCopy':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dCopy_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dCopy_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dCopy(vglCl3dCopy_img_input, vglCl3dCopy_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dCopy(vglCl3dCopy_img_input, vglCl3dCopy_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dCopy: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dCopy_img_output, workspace)


        elif vGlyph.func == 'vglCl3dDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dDilate(vglCl3dDilate_img_input, vglCl3dDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dDilate(vglCl3dDilate_img_input, vglCl3dDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dErode(vglCl3dErode_img_input, vglCl3dErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dErode(vglCl3dErode_img_input, vglCl3dErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dMax':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dMax_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1' , workspace)
          vl.vglCheckContext(vglCl3dMax_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dMax_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2' , workspace)
          vl.vglCheckContext(vglCl3dMax_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dMax_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dMax_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dMax(vglCl3dMax_img_input1, vglCl3dMax_img_input2, vglCl3dMax_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dMax(vglCl3dMax_img_input1, vglCl3dMax_img_input2, vglCl3dMax_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dMax: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dMax_img_output, workspace)


        elif vGlyph.func == 'vglCl3dMin':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dMin_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1' , workspace)
          vl.vglCheckContext(vglCl3dMin_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dMin_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2' , workspace)
          vl.vglCheckContext(vglCl3dMin_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dMin_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dMin_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dMin(vglCl3dMin_img_input1, vglCl3dMin_img_input2, vglCl3dMin_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dMin(vglCl3dMin_img_input1, vglCl3dMin_img_input2, vglCl3dMin_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dMin: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dMin_img_output, workspace)


        elif vGlyph.func == 'vglCl3dNot':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dNot_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dNot_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dNot_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dNot_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dNot(vglCl3dNot_img_input, vglCl3dNot_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dNot(vglCl3dNot_img_input, vglCl3dNot_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dNot: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dNot_img_output, workspace)


        elif vGlyph.func == 'vglCl3dSub':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1' , workspace)
          vl.vglCheckContext(vglCl3dSub_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2' , workspace)
          vl.vglCheckContext(vglCl3dSub_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dSub_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dSub(vglCl3dSub_img_input1, vglCl3dSub_img_input2, vglCl3dSub_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dSub(vglCl3dSub_img_input1, vglCl3dSub_img_input2, vglCl3dSub_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dSub: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dSub_img_output, workspace)


        elif vGlyph.func == 'vglCl3dSum':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dSum_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1' , workspace)
          vl.vglCheckContext(vglCl3dSum_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dSum_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2' , workspace)
          vl.vglCheckContext(vglCl3dSum_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dSum_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dSum_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dSum(vglCl3dSum_img_input1, vglCl3dSum_img_input2, vglCl3dSum_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dSum(vglCl3dSum_img_input1, vglCl3dSum_img_input2, vglCl3dSum_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dSum: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dSum_img_output, workspace)


        elif vGlyph.func == 'vglCl3dThreshold':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dThreshold_src = getImageInputByIdName(vGlyph.glyph_id, 'src' , workspace)
          vl.vglCheckContext(vglCl3dThreshold_src, vl.VGL_CL_CONTEXT());
          vglCl3dThreshold_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst' , workspace)
          vl.vglCheckContext(vglCl3dThreshold_dst, vl.VGL_CL_CONTEXT());
          vglCl3dThreshold(vglCl3dThreshold_src, vglCl3dThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dThreshold(vglCl3dThreshold_src, vglCl3dThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dThreshold: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dThreshold_dst, workspace)


        elif vGlyph.func == 'vglClBlurSq3':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClBlurSq3_img_input, vl.VGL_CL_CONTEXT());
          vglClBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClBlurSq3_img_output, vl.VGL_CL_CONTEXT());
          vglClBlurSq3(vglClBlurSq3_img_input, vglClBlurSq3_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClBlurSq3(vglClBlurSq3_img_input, vglClBlurSq3_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClBlurSq3: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClBlurSq3_img_output, workspace)


        elif vGlyph.func == 'vglClConvolution':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglClConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClConvolution_img_output, vl.VGL_CL_CONTEXT());
          vglClConvolution(vglClConvolution_img_input, vglClConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClConvolution(vglClConvolution_img_input, vglClConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClConvolution: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClConvolution_img_output, workspace)


        elif vGlyph.func == 'vglClCopy':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClCopy_img_input, vl.VGL_CL_CONTEXT());
          vglClCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClCopy_img_output, vl.VGL_CL_CONTEXT());
          vglClCopy(vglClCopy_img_input, vglClCopy_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClCopy(vglClCopy_img_input, vglClCopy_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClCopy: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClCopy_img_output, workspace)


        elif vGlyph.func == 'vglClDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClDilate(vglClDilate_img_input, vglClDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClDilate(vglClDilate_img_input, vglClDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClDilate_img_output, workspace)


        elif vGlyph.func == 'vglClErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClErode_img_input, vl.VGL_CL_CONTEXT());
          vglClErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClErode_img_output, vl.VGL_CL_CONTEXT());
          vglClErode(vglClErode_img_input, vglClErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClErode(vglClErode_img_input, vglClErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClErode_img_output, workspace)


        elif vGlyph.func == 'vglClInvert':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClInvert_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClInvert_img_input, vl.VGL_CL_CONTEXT());
          vglClInvert_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClInvert_img_output, vl.VGL_CL_CONTEXT());
          vglClInvert(vglClInvert_img_input, vglClInvert_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClInvert(vglClInvert_img_input, vglClInvert_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClInvert: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClInvert_img_output, workspace)


        elif vGlyph.func == 'vglClMax':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClMax_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1' , workspace)
          vl.vglCheckContext(vglClMax_img_input1, vl.VGL_CL_CONTEXT());
          vglClMax_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2' , workspace)
          vl.vglCheckContext(vglClMax_img_input2, vl.VGL_CL_CONTEXT());
          vglClMax_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClMax_img_output, vl.VGL_CL_CONTEXT());
          vglClMax(vglClMax_img_input1, vglClMax_img_input2, vglClMax_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClMax(vglClMax_img_input1, vglClMax_img_input2, vglClMax_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClMax: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClMax_img_output, workspace)


        elif vGlyph.func == 'vglClMin':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClMin_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1' , workspace)
          vl.vglCheckContext(vglClMin_img_input1, vl.VGL_CL_CONTEXT());
          vglClMin_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2' , workspace)
          vl.vglCheckContext(vglClMin_img_input2, vl.VGL_CL_CONTEXT());
          vglClMin_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClMin_img_output, vl.VGL_CL_CONTEXT());
          vglClMin(vglClMin_img_input1, vglClMin_img_input2, vglClMin_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClMin(vglClMin_img_input1, vglClMin_img_input2, vglClMin_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClMin: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClMin_img_output, workspace)


        elif vGlyph.func == 'vglClRgb2Gray':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClRgb2Gray_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClRgb2Gray_img_input, vl.VGL_CL_CONTEXT());
          vglClRgb2Gray_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClRgb2Gray_img_output, vl.VGL_CL_CONTEXT());
          vglClRgb2Gray(vglClRgb2Gray_img_input, vglClRgb2Gray_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClRgb2Gray(vglClRgb2Gray_img_input, vglClRgb2Gray_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClRgb2Gray: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClRgb2Gray_img_output, workspace)


        elif vGlyph.func == 'vglClSub':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1' , workspace)
          vl.vglCheckContext(vglClSub_img_input1, vl.VGL_CL_CONTEXT());
          vglClSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2' , workspace)
          vl.vglCheckContext(vglClSub_img_input2, vl.VGL_CL_CONTEXT());
          vglClSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClSub_img_output, vl.VGL_CL_CONTEXT());
          vglClSub(vglClSub_img_input1, vglClSub_img_input2, vglClSub_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClSub(vglClSub_img_input1, vglClSub_img_input2, vglClSub_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClSub: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSub_img_output, workspace)


        elif vGlyph.func == 'vglClSum':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClSum_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1' , workspace)
          vl.vglCheckContext(vglClSum_img_input1, vl.VGL_CL_CONTEXT());
          vglClSum_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2' , workspace)
          vl.vglCheckContext(vglClSum_img_input2, vl.VGL_CL_CONTEXT());
          vglClSum_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClSum_img_output, vl.VGL_CL_CONTEXT());
          vglClSum(vglClSum_img_input1, vglClSum_img_input2, vglClSum_img_output)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClSum(vglClSum_img_input1, vglClSum_img_input2, vglClSum_img_output)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClSum: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSum_img_output, workspace)


        elif vGlyph.func == 'vglClSwapRgb':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClSwapRgb_src = getImageInputByIdName(vGlyph.glyph_id, 'src' , workspace)
          vl.vglCheckContext(vglClSwapRgb_src, vl.VGL_CL_CONTEXT());
          vglClSwapRgb_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst' , workspace)
          vl.vglCheckContext(vglClSwapRgb_dst, vl.VGL_CL_CONTEXT());
          vglClSwapRgb(vglClSwapRgb_src, vglClSwapRgb_dst)

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClSwapRgb(vglClSwapRgb_src, vglClSwapRgb_dst)
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClSwapRgb: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSwapRgb_dst, workspace)


        elif vGlyph.func == 'vglClThreshold':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClThreshold_src = getImageInputByIdName(vGlyph.glyph_id, 'src' , workspace)
          vl.vglCheckContext(vglClThreshold_src, vl.VGL_CL_CONTEXT());
          vglClThreshold_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst' , workspace)
          vl.vglCheckContext(vglClThreshold_dst, vl.VGL_CL_CONTEXT());
          vglClThreshold(vglClThreshold_src, vglClThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClThreshold(vglClThreshold_src, vglClThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClThreshold: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClThreshold_dst, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyAlgDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyAlgDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyAlgDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyAlgDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgDilate(vglCl3dFuzzyAlgDilate_img_input, vglCl3dFuzzyAlgDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyAlgDilate(vglCl3dFuzzyAlgDilate_img_input, vglCl3dFuzzyAlgDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyAlgDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyAlgDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyAlgErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyAlgErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyAlgErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyAlgErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgErode(vglCl3dFuzzyAlgErode_img_input, vglCl3dFuzzyAlgErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyAlgErode(vglCl3dFuzzyAlgErode_img_input, vglCl3dFuzzyAlgErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyAlgErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyAlgErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyArithDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyArithDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyArithDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyArithDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithDilate(vglCl3dFuzzyArithDilate_img_input, vglCl3dFuzzyArithDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyArithDilate(vglCl3dFuzzyArithDilate_img_input, vglCl3dFuzzyArithDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyArithDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyArithDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyArithErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyArithErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyArithErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyArithErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithErode(vglCl3dFuzzyArithErode_img_input, vglCl3dFuzzyArithErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyArithErode(vglCl3dFuzzyArithErode_img_input, vglCl3dFuzzyArithErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyArithErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyArithErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyBoundDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyBoundDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyBoundDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyBoundDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundDilate(vglCl3dFuzzyBoundDilate_img_input, vglCl3dFuzzyBoundDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyBoundDilate(vglCl3dFuzzyBoundDilate_img_input, vglCl3dFuzzyBoundDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyBoundDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyBoundDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyBoundErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyBoundErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyBoundErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyBoundErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundErode(vglCl3dFuzzyBoundErode_img_input, vglCl3dFuzzyBoundErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyBoundErode(vglCl3dFuzzyBoundErode_img_input, vglCl3dFuzzyBoundErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyBoundErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyBoundErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyDaPDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyDaPDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyDaPDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyDaPDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPDilate(vglCl3dFuzzyDaPDilate_img_input, vglCl3dFuzzyDaPDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyDaPDilate(vglCl3dFuzzyDaPDilate_img_input, vglCl3dFuzzyDaPDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyDaPDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDaPDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyDaPErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyDaPErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyDaPErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyDaPErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPErode(vglCl3dFuzzyDaPErode_img_input, vglCl3dFuzzyDaPErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyDaPErode(vglCl3dFuzzyDaPErode_img_input, vglCl3dFuzzyDaPErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyDaPErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDaPErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyDrasticDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyDrasticDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyDrasticDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyDrasticDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticDilate(vglCl3dFuzzyDrasticDilate_img_input, vglCl3dFuzzyDrasticDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyDrasticDilate(vglCl3dFuzzyDrasticDilate_img_input, vglCl3dFuzzyDrasticDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyDrasticDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDrasticDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyDrasticErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyDrasticErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyDrasticErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyDrasticErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticErode(vglCl3dFuzzyDrasticErode_img_input, vglCl3dFuzzyDrasticErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyDrasticErode(vglCl3dFuzzyDrasticErode_img_input, vglCl3dFuzzyDrasticErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyDrasticErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDrasticErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyGeoDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyGeoDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyGeoDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyGeoDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoDilate(vglCl3dFuzzyGeoDilate_img_input, vglCl3dFuzzyGeoDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyGeoDilate(vglCl3dFuzzyGeoDilate_img_input, vglCl3dFuzzyGeoDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyGeoDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyGeoDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyGeoErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyGeoErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyGeoErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyGeoErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoErode(vglCl3dFuzzyGeoErode_img_input, vglCl3dFuzzyGeoErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyGeoErode(vglCl3dFuzzyGeoErode_img_input, vglCl3dFuzzyGeoErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyGeoErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyGeoErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyHamacherDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyHamacherDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyHamacherDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyHamacherDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherDilate(vglCl3dFuzzyHamacherDilate_img_input, vglCl3dFuzzyHamacherDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyHamacherDilate(vglCl3dFuzzyHamacherDilate_img_input, vglCl3dFuzzyHamacherDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyHamacherDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyHamacherDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyHamacherErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyHamacherErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyHamacherErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyHamacherErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherErode(vglCl3dFuzzyHamacherErode_img_input, vglCl3dFuzzyHamacherErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyHamacherErode(vglCl3dFuzzyHamacherErode_img_input, vglCl3dFuzzyHamacherErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyHamacherErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyHamacherErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyStdDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyStdDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyStdDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyStdDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdDilate(vglCl3dFuzzyStdDilate_img_input, vglCl3dFuzzyStdDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyStdDilate(vglCl3dFuzzyStdDilate_img_input, vglCl3dFuzzyStdDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyStdDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyStdDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyStdErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglCl3dFuzzyStdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyStdErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglCl3dFuzzyStdErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdErode(vglCl3dFuzzyStdErode_img_input, vglCl3dFuzzyStdErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglCl3dFuzzyStdErode(vglCl3dFuzzyStdErode_img_input, vglCl3dFuzzyStdErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglCl3dFuzzyStdErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyStdErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyAlgDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyAlgDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyAlgDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyAlgDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgDilate(vglClFuzzyAlgDilate_img_input, vglClFuzzyAlgDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyAlgDilate(vglClFuzzyAlgDilate_img_input, vglClFuzzyAlgDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyAlgDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyAlgDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyAlgErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyAlgErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyAlgErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyAlgErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgErode(vglClFuzzyAlgErode_img_input, vglClFuzzyAlgErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyAlgErode(vglClFuzzyAlgErode_img_input, vglClFuzzyAlgErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyAlgErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyAlgErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyArithDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyArithDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyArithDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyArithDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithDilate(vglClFuzzyArithDilate_img_input, vglClFuzzyArithDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyArithDilate(vglClFuzzyArithDilate_img_input, vglClFuzzyArithDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyArithDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyArithDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyArithErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyArithErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyArithErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyArithErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithErode(vglClFuzzyArithErode_img_input, vglClFuzzyArithErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyArithErode(vglClFuzzyArithErode_img_input, vglClFuzzyArithErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyArithErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyArithErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyBoundDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyBoundDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyBoundDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyBoundDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundDilate(vglClFuzzyBoundDilate_img_input, vglClFuzzyBoundDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyBoundDilate(vglClFuzzyBoundDilate_img_input, vglClFuzzyBoundDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyBoundDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyBoundDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyBoundErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyBoundErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyBoundErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyBoundErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundErode(vglClFuzzyBoundErode_img_input, vglClFuzzyBoundErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyBoundErode(vglClFuzzyBoundErode_img_input, vglClFuzzyBoundErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyBoundErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyBoundErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyDaPDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyDaPDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyDaPDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyDaPDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPDilate(vglClFuzzyDaPDilate_img_input, vglClFuzzyDaPDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyDaPDilate(vglClFuzzyDaPDilate_img_input, vglClFuzzyDaPDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyDaPDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDaPDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyDaPErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyDaPErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyDaPErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyDaPErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPErode(vglClFuzzyDaPErode_img_input, vglClFuzzyDaPErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyDaPErode(vglClFuzzyDaPErode_img_input, vglClFuzzyDaPErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyDaPErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDaPErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyDrasticDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyDrasticDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyDrasticDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyDrasticDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticDilate(vglClFuzzyDrasticDilate_img_input, vglClFuzzyDrasticDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyDrasticDilate(vglClFuzzyDrasticDilate_img_input, vglClFuzzyDrasticDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyDrasticDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDrasticDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyDrasticErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyDrasticErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyDrasticErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyDrasticErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticErode(vglClFuzzyDrasticErode_img_input, vglClFuzzyDrasticErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyDrasticErode(vglClFuzzyDrasticErode_img_input, vglClFuzzyDrasticErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyDrasticErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDrasticErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyGeoDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyGeoDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyGeoDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyGeoDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoDilate(vglClFuzzyGeoDilate_img_input, vglClFuzzyGeoDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyGeoDilate(vglClFuzzyGeoDilate_img_input, vglClFuzzyGeoDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyGeoDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyGeoDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyGeoErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyGeoErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyGeoErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyGeoErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoErode(vglClFuzzyGeoErode_img_input, vglClFuzzyGeoErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyGeoErode(vglClFuzzyGeoErode_img_input, vglClFuzzyGeoErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyGeoErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyGeoErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyHamacherDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyHamacherDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyHamacherDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyHamacherDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherDilate(vglClFuzzyHamacherDilate_img_input, vglClFuzzyHamacherDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyHamacherDilate(vglClFuzzyHamacherDilate_img_input, vglClFuzzyHamacherDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyHamacherDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyHamacherDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyHamacherErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyHamacherErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyHamacherErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyHamacherErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherErode(vglClFuzzyHamacherErode_img_input, vglClFuzzyHamacherErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyHamacherErode(vglClFuzzyHamacherErode_img_input, vglClFuzzyHamacherErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyHamacherErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyHamacherErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyStdDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyStdDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyStdDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyStdDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdDilate(vglClFuzzyStdDilate_img_input, vglClFuzzyStdDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyStdDilate(vglClFuzzyStdDilate_img_input, vglClFuzzyStdDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyStdDilate: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyStdDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyStdErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")
          print("-------------------------------------------------")


          vglClFuzzyStdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input' , workspace)
          vl.vglCheckContext(vglClFuzzyStdErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output' , workspace)
          vl.vglCheckContext(vglClFuzzyStdErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdErode(vglClFuzzyStdErode_img_input, vglClFuzzyStdErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          # Runtime
          t0 = datetime.now()
          for i in range(nSteps):
            vglClFuzzyStdErode(vglClFuzzyStdErode_img_input, vglClFuzzyStdErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          t1 = datetime.now()
          t = t1 - t0
          media = round((t.total_seconds() * 1000) / nSteps, 3)
          msg = msg + "Tempo médio de " + str(nSteps) + " execuções do método vglClFuzzyStdErode: " + str(media) + " ms\n"
          total = total + media


          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyStdErode_img_output, workspace)

    print(msg)
    print("-------------------------------------------------------------")
    print("O valor total do tempo médio : "+str(round(total, 3)) , "em ms" )
    print("-------------------------------------------------------------")
execute_workspace(workspace)
