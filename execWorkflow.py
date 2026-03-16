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
workspace = Workspace()

fileRead(workspace)

# Actions after glyph execution
def GlyphExecutedUpdate(GlyphExecutedUpdate_Glyph_Id, GlyphExecutedUpdate_image, workspace):
    # Rule10: Glyph becomes DONE = TRUE after its execution. Assign done to glyph
    setGlyphDoneId(GlyphExecutedUpdate_Glyph_Id,workspace)

    # Rule6: Edges whose source glyph has already been executed, and which therefore already had their image generated, have READY=TRUE (image ready to be processed).
    #        Reading the image from another glyph does not change this status. Check the list of connections
    setGlyphInputReadyByIdOut(GlyphExecutedUpdate_Glyph_Id, workspace)

    # Rule2: In a source glyph, images (one or more) can only be output parameters.
    setImageConnectionByOutputId(GlyphExecutedUpdate_Glyph_Id, GlyphExecutedUpdate_image, workspace)
    
def execWorkflow(workspace, is_subworkflow=False, parent_workflow_id=None, processed_workflows=None):
    """
    Executa os glyphs e sub-workflows presentes no workspace.
    """
    if processed_workflows is None:
        processed_workflows = set()

    for vGlyph in workspace.lstGlyph:
        # Evita processar glyphs já executados
        if vGlyph.glyph_id in processed_workflows:
            continue

        # Processa sub-workspaces diretamente
        if hasattr(vGlyph, "sub_workspaces"):  # Verifica se o glyph contém sub-workspaces
            for sub_workspace in vGlyph.sub_workspaces:
                print(f"Iniciando sub-workflow (ID: {vGlyph.glyph_id})...")
                execWorkflow(sub_workspace, is_subworkflow=True, parent_workflow_id=vGlyph.glyph_id, processed_workflows=processed_workflows)
            continue

        
        if vGlyph.func in ('vglLoad2dImage', 'vglLoadImage'):
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

        elif vGlyph.func == 'vglCreateImage':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

            vglCreateImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img', workspace)
            vglCreateImage_RETVAL = vl.create_blank_image_as(vglCreateImage_img_input)
            vglCreateImage_RETVAL.set_oclPtr(vl.get_similar_oclPtr_object(vglCreateImage_img_input))
            vl.vglAddContext(vglCreateImage_RETVAL, vl.VGL_CL_CONTEXT())
            GlyphExecutedUpdate(vGlyph.glyph_id, vglCreateImage_RETVAL, workspace)



        elif vGlyph.func == 'External Output (1)':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            o = getImageInputByIdName(vGlyph.glyph_id, 'o', workspace)
            GlyphExecutedUpdate(vGlyph.glyph_id, o, workspace)

        elif vGlyph.func == 'External Input (1)':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            o = getImageInputByIdName(vGlyph.glyph_id, 'i', workspace)
            GlyphExecutedUpdate(vGlyph.glyph_id, o, workspace)

        elif vGlyph.func == 'ProcedureBegin':
            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")

            proc_name = vGlyph.library
            sub_ws = next(
                (s for s in workspace.subWorkspaces if getattr(s, 'name', None) == proc_name),
                None
            )
            if sub_ws is None and workspace.subWorkspaces:
                sub_ws = workspace.subWorkspaces[0]

            input_img = getImageInputByIdName(vGlyph.glyph_id, 'i', workspace)

            if sub_ws is not None and input_img is not None:
                ext_in = next(
                    (g for g in sub_ws.lstGlyph if g.func == "External Input (1)"),
                    None
                )
                if ext_in:
                    GlyphExecutedUpdate(ext_in.glyph_id, input_img, sub_ws)

                execWorkflow(sub_ws, is_subworkflow=True,
                             parent_workflow_id=vGlyph.glyph_id,
                             processed_workflows=processed_workflows)

                ext_out = next(
                    (g for g in sub_ws.lstGlyph if g.func == "External Output (1)"),
                    None
                )
                o = None
                if ext_out:
                    o = getImageInputByIdName(ext_out.glyph_id, 'o', sub_ws)
                GlyphExecutedUpdate(vGlyph.glyph_id, o, workspace)
            else:
                print(f"Aviso: procedure '{proc_name}' sem imagem de entrada ou sub-workspace não encontrado.")
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


            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, Rec_img_output, workspace)

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

        elif vGlyph.func == 'ShowImage':

            print("-------------------------------------------------")
            print("A função " + vGlyph.func + " está sendo executada")
            print("-------------------------------------------------")


            # Returns edge image based on glyph id
            ShowImage_img_input = getImageInputByIdName(vGlyph.glyph_id, 'image', workspace)

            if ShowImage_img_input is not None:
                # Rule3: In a sink glyph, images (one or more) can only be input parameters
                vl.vglCheckContext(ShowImage_img_input, vl.VGL_RAM_CONTEXT())
                ShowImage_img_ndarray = VglImage.get_ipl(ShowImage_img_input)

                # Salva em arquivo temporário e emite marcador para a GUI exibir preview
                try:
                    import tempfile
                    _tmp = tempfile.mktemp(suffix='.png')
                    mp.imsave(_tmp, ShowImage_img_ndarray, cmap='gray')
                    print(f'[GUI_SHOW] {_tmp}')
                    sys.stdout.flush()
                except Exception as _e:
                    print(f'[ShowImage] Erro ao salvar preview: {_e}')

                # Actions after glyph execution
                GlyphExecutedUpdate(vGlyph.glyph_id, None, workspace)

        elif vGlyph.func == 'vglClNdConvolution':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClNdConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClNdConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglClNdConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClNdConvolution_img_output, vl.VGL_CL_CONTEXT());
          window = getImageInputByIdName(vGlyph.glyph_id, 'window', workspace)
          vglClNdConvolution(vglClNdConvolution_img_input, vglClNdConvolution_img_output, window)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdConvolution_img_output, workspace)


        elif vGlyph.func == 'vglClNdCopy':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClNdCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClNdCopy_img_input, vl.VGL_CL_CONTEXT());
          vglClNdCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClNdCopy_img_output, vl.VGL_CL_CONTEXT());
          vglClNdCopy(vglClNdCopy_img_input, vglClNdCopy_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdCopy_img_output, workspace)


        elif vGlyph.func == 'vglClNdDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClNdDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClNdDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClNdDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClNdDilate_img_output, vl.VGL_CL_CONTEXT());
          window = getImageInputByIdName(vGlyph.glyph_id, 'window', workspace)
          vglClNdDilate(vglClNdDilate_img_input, vglClNdDilate_img_output, window)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdDilate_img_output, workspace)


        elif vGlyph.func == 'vglClNdErode': #Function Erode
            print("-------------------------------------------------")
            print("A função " + vGlyph.func +" está sendo executada")
            print("-------------------------------------------------")

            # Search the input image by connecting to the source glyph
            vglClNdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
            
            # Search the output image by connecting to the source glyph
            vglClNdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
            
            # Apply Erode function
            vl.vglCheckContext(vglClNdErode_img_output,vl.VGL_RAM_CONTEXT())

            window = getImageInputByIdName(vGlyph.glyph_id, 'window', workspace)

            vglClNdErode(vglClNdErode_img_input, vglClNdErode_img_output, window)
            
            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdErode_img_output, workspace)



        elif vGlyph.func == 'vglClNdNot':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClNdNot_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClNdNot_img_input, vl.VGL_CL_CONTEXT());
          vglClNdNot_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClNdNot_img_output, vl.VGL_CL_CONTEXT());
          vglClNdNot(vglClNdNot_img_input, vglClNdNot_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdNot_img_output, workspace)


        elif vGlyph.func == 'vglClNdThreshold':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClNdThreshold_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClNdThreshold_img_input, vl.VGL_CL_CONTEXT());
          vglClNdThreshold_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClNdThreshold_img_output, vl.VGL_CL_CONTEXT());
          vglClNdThreshold(vglClNdThreshold_img_input, vglClNdThreshold_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClNdThreshold_img_output, workspace)


        elif vGlyph.func == 'vglCl3dBlurSq3':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dBlurSq3_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dBlurSq3_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dBlurSq3(vglCl3dBlurSq3_img_input, vglCl3dBlurSq3_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dBlurSq3_img_output, workspace)


        elif vGlyph.func == 'vglCl3dConvolution':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dConvolution_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dConvolution(vglCl3dConvolution_img_input, vglCl3dConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dConvolution_img_output, workspace)


        elif vGlyph.func == 'vglCl3dCopy':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dCopy_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dCopy_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dCopy(vglCl3dCopy_img_input, vglCl3dCopy_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dCopy_img_output, workspace)


        elif vGlyph.func == 'vglCl3dDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dDilate(vglCl3dDilate_img_input, vglCl3dDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dErode(vglCl3dErode_img_input, vglCl3dErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dMax':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dMax_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1', workspace)
          vl.vglCheckContext(vglCl3dMax_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dMax_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2', workspace)
          vl.vglCheckContext(vglCl3dMax_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dMax_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dMax_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dMax(vglCl3dMax_img_input1, vglCl3dMax_img_input2, vglCl3dMax_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dMax_img_output, workspace)


        elif vGlyph.func == 'vglCl3dMin':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dMin_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1', workspace)
          vl.vglCheckContext(vglCl3dMin_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dMin_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2', workspace)
          vl.vglCheckContext(vglCl3dMin_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dMin_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dMin_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dMin(vglCl3dMin_img_input1, vglCl3dMin_img_input2, vglCl3dMin_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dMin_img_output, workspace)


        elif vGlyph.func == 'vglCl3dNot':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dNot_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dNot_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dNot_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dNot_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dNot(vglCl3dNot_img_input, vglCl3dNot_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dNot_img_output, workspace)


        elif vGlyph.func == 'vglCl3dSub':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1', workspace)
          vl.vglCheckContext(vglCl3dSub_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2', workspace)
          vl.vglCheckContext(vglCl3dSub_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dSub_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dSub(vglCl3dSub_img_input1, vglCl3dSub_img_input2, vglCl3dSub_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dSub_img_output, workspace)


        elif vGlyph.func == 'vglCl3dSum':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dSum_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1', workspace)
          vl.vglCheckContext(vglCl3dSum_img_input1, vl.VGL_CL_CONTEXT());
          vglCl3dSum_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2', workspace)
          vl.vglCheckContext(vglCl3dSum_img_input2, vl.VGL_CL_CONTEXT());
          vglCl3dSum_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dSum_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dSum(vglCl3dSum_img_input1, vglCl3dSum_img_input2, vglCl3dSum_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dSum_img_output, workspace)


        elif vGlyph.func == 'vglCl3dThreshold':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dThreshold_src = getImageInputByIdName(vGlyph.glyph_id, 'src', workspace)
          vl.vglCheckContext(vglCl3dThreshold_src, vl.VGL_CL_CONTEXT());
          vglCl3dThreshold_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst', workspace)
          vl.vglCheckContext(vglCl3dThreshold_dst, vl.VGL_CL_CONTEXT());
          vglCl3dThreshold(vglCl3dThreshold_src, vglCl3dThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dThreshold_dst, workspace)


        elif vGlyph.func == 'vglClBlurSq3':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClBlurSq3_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClBlurSq3_img_input, vl.VGL_CL_CONTEXT());
          vglClBlurSq3_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClBlurSq3_img_output, vl.VGL_CL_CONTEXT());
          vglClBlurSq3(vglClBlurSq3_img_input, vglClBlurSq3_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClBlurSq3_img_output, workspace)


        elif vGlyph.func == 'vglClConvolution':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClConvolution_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClConvolution_img_input, vl.VGL_CL_CONTEXT());
          vglClConvolution_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClConvolution_img_output, vl.VGL_CL_CONTEXT());
          vglClConvolution(vglClConvolution_img_input, vglClConvolution_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClConvolution_img_output, workspace)


        elif vGlyph.func == 'vglClCopy':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClCopy_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClCopy_img_input, vl.VGL_CL_CONTEXT());
          vglClCopy_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClCopy_img_output, vl.VGL_CL_CONTEXT());
          vglClCopy(vglClCopy_img_input, vglClCopy_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClCopy_img_output, workspace)


        elif vGlyph.func == 'vglClDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClDilate(vglClDilate_img_input, vglClDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClDilate_img_output, workspace)


        elif vGlyph.func == 'vglClErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func +" está sendo executada")
          
          vglClErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          print(f"Imagem após vglClErode: {vglClErode_img_input.shape}")


          vl.vglCheckContext(vglClErode_img_input, vl.VGL_CL_CONTEXT());
          vglClErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClErode_img_output, vl.VGL_CL_CONTEXT());
          vglClErode(vglClErode_img_input, vglClErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))
          
          print(f"Imagem após vglClErode: {vglClErode_img_output.shape}")
          GlyphExecutedUpdate(vGlyph.glyph_id, vglClErode_img_output, workspace)


        elif vGlyph.func == 'vglClInvert':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClInvert_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClInvert_img_input, vl.VGL_CL_CONTEXT());
          vglClInvert_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClInvert_img_output, vl.VGL_CL_CONTEXT());
          vglClInvert(vglClInvert_img_input, vglClInvert_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClInvert_img_output, workspace)


        elif vGlyph.func == 'vglClMax':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClMax_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1', workspace)
          vl.vglCheckContext(vglClMax_img_input1, vl.VGL_CL_CONTEXT());
          vglClMax_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2', workspace)
          vl.vglCheckContext(vglClMax_img_input2, vl.VGL_CL_CONTEXT());
          vglClMax_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClMax_img_output, vl.VGL_CL_CONTEXT());
          vglClMax(vglClMax_img_input1, vglClMax_img_input2, vglClMax_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClMax_img_output, workspace)


        elif vGlyph.func == 'vglClMin':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClMin_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1', workspace)
          vl.vglCheckContext(vglClMin_img_input1, vl.VGL_CL_CONTEXT());
          vglClMin_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2', workspace)
          vl.vglCheckContext(vglClMin_img_input2, vl.VGL_CL_CONTEXT());
          vglClMin_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClMin_img_output, vl.VGL_CL_CONTEXT());
          vglClMin(vglClMin_img_input1, vglClMin_img_input2, vglClMin_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClMin_img_output, workspace)


        elif vGlyph.func == 'vglClRgb2Gray':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClRgb2Gray_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClRgb2Gray_img_input, vl.VGL_CL_CONTEXT());
          vglClRgb2Gray_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClRgb2Gray_img_output, vl.VGL_CL_CONTEXT());
          vglClRgb2Gray(vglClRgb2Gray_img_input, vglClRgb2Gray_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClRgb2Gray_img_output, workspace)


        elif vGlyph.func == 'vglClSub':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClSub_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1', workspace)
          vl.vglCheckContext(vglClSub_img_input1, vl.VGL_CL_CONTEXT());
          vglClSub_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2', workspace)
          vl.vglCheckContext(vglClSub_img_input2, vl.VGL_CL_CONTEXT());
          vglClSub_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClSub_img_output, vl.VGL_CL_CONTEXT());
          vglClSub(vglClSub_img_input1, vglClSub_img_input2, vglClSub_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSub_img_output, workspace)


        elif vGlyph.func == 'vglClSum':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClSum_img_input1 = getImageInputByIdName(vGlyph.glyph_id, 'img_input1', workspace)
          vl.vglCheckContext(vglClSum_img_input1, vl.VGL_CL_CONTEXT());
          vglClSum_img_input2 = getImageInputByIdName(vGlyph.glyph_id, 'img_input2', workspace)
          vl.vglCheckContext(vglClSum_img_input2, vl.VGL_CL_CONTEXT());
          vglClSum_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClSum_img_output, vl.VGL_CL_CONTEXT());
          vglClSum(vglClSum_img_input1, vglClSum_img_input2, vglClSum_img_output)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSum_img_output, workspace)


        elif vGlyph.func == 'vglClSwapRgb':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClSwapRgb_src = getImageInputByIdName(vGlyph.glyph_id, 'src', workspace)
          vl.vglCheckContext(vglClSwapRgb_src, vl.VGL_CL_CONTEXT());
          vglClSwapRgb_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst', workspace)
          vl.vglCheckContext(vglClSwapRgb_dst, vl.VGL_CL_CONTEXT());
          vglClSwapRgb(vglClSwapRgb_src, vglClSwapRgb_dst)

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClSwapRgb_dst, workspace)


        elif vGlyph.func == 'vglClThreshold':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClThreshold_src = getImageInputByIdName(vGlyph.glyph_id, 'src', workspace)
          vl.vglCheckContext(vglClThreshold_src, vl.VGL_CL_CONTEXT());
          vglClThreshold_dst = getImageInputByIdName(vGlyph.glyph_id, 'dst', workspace)
          vl.vglCheckContext(vglClThreshold_dst, vl.VGL_CL_CONTEXT());
          vglClThreshold(vglClThreshold_src, vglClThreshold_dst, np.float32(vGlyph.lst_par[0].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClThreshold_dst, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyAlgDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyAlgDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyAlgDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyAlgDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgDilate(vglCl3dFuzzyAlgDilate_img_input, vglCl3dFuzzyAlgDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyAlgDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyAlgErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyAlgErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyAlgErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyAlgErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyAlgErode(vglCl3dFuzzyAlgErode_img_input, vglCl3dFuzzyAlgErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyAlgErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyArithDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyArithDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyArithDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyArithDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithDilate(vglCl3dFuzzyArithDilate_img_input, vglCl3dFuzzyArithDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyArithDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyArithErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyArithErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyArithErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyArithErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyArithErode(vglCl3dFuzzyArithErode_img_input, vglCl3dFuzzyArithErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyArithErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyBoundDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyBoundDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyBoundDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyBoundDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundDilate(vglCl3dFuzzyBoundDilate_img_input, vglCl3dFuzzyBoundDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyBoundDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyBoundErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyBoundErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyBoundErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyBoundErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyBoundErode(vglCl3dFuzzyBoundErode_img_input, vglCl3dFuzzyBoundErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyBoundErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyDaPDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyDaPDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyDaPDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyDaPDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPDilate(vglCl3dFuzzyDaPDilate_img_input, vglCl3dFuzzyDaPDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDaPDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyDaPErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyDaPErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyDaPErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyDaPErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDaPErode(vglCl3dFuzzyDaPErode_img_input, vglCl3dFuzzyDaPErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDaPErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyDrasticDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyDrasticDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyDrasticDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyDrasticDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticDilate(vglCl3dFuzzyDrasticDilate_img_input, vglCl3dFuzzyDrasticDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDrasticDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyDrasticErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyDrasticErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyDrasticErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyDrasticErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyDrasticErode(vglCl3dFuzzyDrasticErode_img_input, vglCl3dFuzzyDrasticErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyDrasticErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyGeoDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyGeoDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyGeoDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyGeoDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoDilate(vglCl3dFuzzyGeoDilate_img_input, vglCl3dFuzzyGeoDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyGeoDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyGeoErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyGeoErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyGeoErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyGeoErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyGeoErode(vglCl3dFuzzyGeoErode_img_input, vglCl3dFuzzyGeoErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyGeoErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyHamacherDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyHamacherDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyHamacherDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyHamacherDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherDilate(vglCl3dFuzzyHamacherDilate_img_input, vglCl3dFuzzyHamacherDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyHamacherDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyHamacherErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyHamacherErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyHamacherErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyHamacherErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyHamacherErode(vglCl3dFuzzyHamacherErode_img_input, vglCl3dFuzzyHamacherErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyHamacherErode_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyStdDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyStdDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyStdDilate_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyStdDilate_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdDilate(vglCl3dFuzzyStdDilate_img_input, vglCl3dFuzzyStdDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyStdDilate_img_output, workspace)


        elif vGlyph.func == 'vglCl3dFuzzyStdErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglCl3dFuzzyStdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglCl3dFuzzyStdErode_img_input, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglCl3dFuzzyStdErode_img_output, vl.VGL_CL_CONTEXT());
          vglCl3dFuzzyStdErode(vglCl3dFuzzyStdErode_img_input, vglCl3dFuzzyStdErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()), np.uint32(vGlyph.lst_par[3].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglCl3dFuzzyStdErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyAlgDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyAlgDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyAlgDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyAlgDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgDilate(vglClFuzzyAlgDilate_img_input, vglClFuzzyAlgDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyAlgDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyAlgErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyAlgErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyAlgErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyAlgErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyAlgErode(vglClFuzzyAlgErode_img_input, vglClFuzzyAlgErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyAlgErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyArithDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyArithDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyArithDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyArithDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithDilate(vglClFuzzyArithDilate_img_input, vglClFuzzyArithDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyArithDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyArithErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyArithErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyArithErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyArithErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyArithErode(vglClFuzzyArithErode_img_input, vglClFuzzyArithErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyArithErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyBoundDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyBoundDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyBoundDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyBoundDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundDilate(vglClFuzzyBoundDilate_img_input, vglClFuzzyBoundDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyBoundDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyBoundErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyBoundErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyBoundErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyBoundErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyBoundErode(vglClFuzzyBoundErode_img_input, vglClFuzzyBoundErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyBoundErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyDaPDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyDaPDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyDaPDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyDaPDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPDilate(vglClFuzzyDaPDilate_img_input, vglClFuzzyDaPDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDaPDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyDaPErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyDaPErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyDaPErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyDaPErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDaPErode(vglClFuzzyDaPErode_img_input, vglClFuzzyDaPErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDaPErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyDrasticDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyDrasticDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyDrasticDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyDrasticDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticDilate(vglClFuzzyDrasticDilate_img_input, vglClFuzzyDrasticDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDrasticDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyDrasticErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyDrasticErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyDrasticErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyDrasticErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyDrasticErode(vglClFuzzyDrasticErode_img_input, vglClFuzzyDrasticErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyDrasticErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyGeoDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyGeoDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyGeoDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyGeoDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoDilate(vglClFuzzyGeoDilate_img_input, vglClFuzzyGeoDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyGeoDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyGeoErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyGeoErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyGeoErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyGeoErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyGeoErode(vglClFuzzyGeoErode_img_input, vglClFuzzyGeoErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyGeoErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyHamacherDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyHamacherDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyHamacherDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyHamacherDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherDilate(vglClFuzzyHamacherDilate_img_input, vglClFuzzyHamacherDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyHamacherDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyHamacherErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyHamacherErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyHamacherErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyHamacherErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyHamacherErode(vglClFuzzyHamacherErode_img_input, vglClFuzzyHamacherErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyHamacherErode_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyStdDilate':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyStdDilate_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyStdDilate_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdDilate_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyStdDilate_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdDilate(vglClFuzzyStdDilate_img_input, vglClFuzzyStdDilate_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyStdDilate_img_output, workspace)


        elif vGlyph.func == 'vglClFuzzyStdErode':
          print("-------------------------------------------------")
          print("A função " + vGlyph.func + " está sendo executada")

          vglClFuzzyStdErode_img_input = getImageInputByIdName(vGlyph.glyph_id, 'img_input', workspace)
          vl.vglCheckContext(vglClFuzzyStdErode_img_input, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdErode_img_output = getImageInputByIdName(vGlyph.glyph_id, 'img_output', workspace)
          vl.vglCheckContext(vglClFuzzyStdErode_img_output, vl.VGL_CL_CONTEXT());
          vglClFuzzyStdErode(vglClFuzzyStdErode_img_input, vglClFuzzyStdErode_img_output, tratnum(vGlyph.lst_par[0].getValue()), np.uint32(vGlyph.lst_par[1].getValue()), np.uint32(vGlyph.lst_par[2].getValue()))

          GlyphExecutedUpdate(vGlyph.glyph_id, vglClFuzzyStdErode_img_output, workspace)


execWorkflow(workspace)