from skimage import io
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
import vgl_lib as vl

"""
  img:
    is the input image
  cl_shape:
    3D Images:
      The OpenCL's default is to be (img_width, img_Heigth, img_depth)
    2D Images:
      The The OpenCL's default is to be (img_width, img_Heigth)
  cl_pitch:
    3D Images (needed):
      The OpenCL's default is to be (img_width*bytes_per_pixel, img_Heigth*img_width*bytes_per_pixel)
      and it is assumed when pitch=(0, 0) is given
    2D Images (optional):
      The OpenCL's default is to be (img_width*bytes_per_pixel)
      and it is assumed when pitch=(0) is given
  cl_origin
    Is the origin of the image, where to start copying the image.
    2D images must have 0 in the z-axis
  cl_region
    Is where to end the copying of the image.
    2D images must have 1 in the z-axis
"""

"""
  VglImage is equivalent to vglImage object.
  isBuffer is a inicialization option that sets the
  clForceAsBuf. If it is not set in the image creation,
  is assumed that is a OpenCL Image object.

  ndim sets the number of dimensions of the image. If it
  is not set un the image creation, is assumed that is a
  2D image.

  ALL THE CONSTANTS CAN BE FOUND IN vglConst.py file.

  What python-version VglImage don't have right now?
    GLuint fbo equivalent; (GL not implemented)
    GLuint tex equivalent; (GL not implemented)
    void* cudaPtr equivalent; (CUDA not implemented)
    GLuint cudaPbo equivalent; (CUDA not implemented)
"""
class VglImage(object):
  def __init__(self, imgPath="", depth=None, ndim=None, clForceAsBuf=None ):
    # IF THE IMAGE TYPE IS NOT SPECIFIED, A 2D IMAGE WILL BE ASSUMED
    # INICIALIZING DATA
    self.ipl = None
    self.ndim = ndim
    self.shape = np.zeros((2*vl.VGL_MAX_DIM()), np.uint8)
    self.vglShape = None
    self.depth = depth
    self.nChannels = 0
    self.has_mipmap = 0
    self.oclPtr = None
    self.clForceAsBuf = clForceAsBuf
    self.inContext = 0
    self.filename = imgPath

    # NOT IMPLEMENTED IN PYTHON-SIDE
    self.fbo = -1
    self.tex = -1

    self.cudaPtr = None
    self.cudaPbo = -1

    if( self.depth == None ):
      self.depth = vl.IPL_DEPTH_1U()

    if( self.clForceAsBuf is None ):
      self.clForceAsBuf = vl.IMAGE_CL_OBJECT()
    elif( not((self.clForceAsBuf is vl.IMAGE_CL_OBJECT() )
         or (self.clForceAsBuf is vl.IMAGE_ND_ARRAY() ) ) ):
      #print("VglImage: Error! Unexistent image treatment. Use vl.IMAGE_CL_OBJECT() or vl.IMAGE_ND_ARRAY()!")
      exit()

    if(self.ndim is None):
      self.ndim = vl.VGL_IMAGE_2D_IMAGE()
      print(":Assuming 2D Image!")
    elif(self.ndim is vl.VGL_IMAGE_2D_IMAGE()):
      print(":Creating 2D Image!")
    elif(self.ndim is vl.VGL_IMAGE_3D_IMAGE()):
      print(":Creating 3D Image!")
    else:
      print("vglImage: Warning! Image is not 2D or 3D. Execution will continue.")
    
    #print(":::-->path", imgPath)
    #print(":::-->dept", depth)
    #print(":::-->ndim", ndim)
    #print(":::-->forc", clForceAsBuf)

  def prinfInfo(self): #printInfo
    print("=========")
    print("VglImage:")
    print("  ipl          = " + str(self.ipl))
    print("  ndim         = " + str(self.ndim))
    print("  shape        = " + str(self.shape))
    self.vglShape.printInfo()
    print("  depth        = " + str(self.depth))
    print("  nChannels    = " + str(self.nChannels))
    print("  has_mipmap   = " + str(self.has_mipmap))
    print("  oclPtr       = " + str(self.oclPtr))
    # if (self.oclPtr is not None):
    #   print("    oclPtr.shape        = " + str(self.oclPtr.shape))
    #   print("    oclPtr.element_size = " + str(self.oclPtr.element_size))
    #   print("    oclPtr.format       = " + str(self.oclPtr.format))
    #   print("    oclPtr.context      = " + str(self.oclPtr.context))
    #   print("    oclPtr.flags        = " + str(self.oclPtr.flags) + " = " + cl.mem_flags.to_string(self.oclPtr.flags) )
    #   print("    is cl.Image         = " + str(isinstance(self.oclPtr, cl.Image)))
    #   print("    is cl.Buffer        = " + str(isinstance(self.oclPtr, cl.Buffer)))
    print("  clForceAsBuf = " + str(self.clForceAsBuf))
    print("  inContext    = " + str(self.inContext))
    print("  filename     = " + str(self.filename))


  def getVglShape(self):
    return self.vglShape

  def set_oclPtr(self, img):
    if( self.clForceAsBuf == vl.IMAGE_CL_OBJECT() ):
      if( isinstance(img, cl.Image) ):
        self.oclPtr = img
      else:
        print("vglImage: set_oclPtr Error! This image must have a OpenCL Image object as oclPtr.")
        print("Image sent:")
        print(img)
        exit()

    elif( self.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
      if( isinstance(img, cl.Buffer) ):
        self.oclPtr = img
      else:
        print("vglImage: set_oclPtr Error: This image must have a OpenCL Buffer object as oclPtr.")
        exit()
    else:
      print("vglImage: set_oclPtr Error: Invalid object. oclPtr must be cl.Image or cl.Buffer objects, according to clForceAsBuf.")
      exit()
    
  def get_oclPtr(self):
    return self.oclPtr
    
  def get_ipl(self):
    return self.ipl
  
  def getNChannels(self):
    return self.vglShape.shape[vl.VGL_SHAPE_NCHANNELS()]
  
  def getWidth(self):
    return self.vglShape.getWidth()
  
  def getHeigth(self):
    return self.vglShape.getHeigth()
  
  def getLength(self):
    return self.vglShape.getLength()
  
  def getWidthIn(self):
    return self.vglShape.getWidthIn()
  
  def getHeigthIn(self):
    return self.vglShape.getHeigthIn()
  
  def getNFrames(self):
    return self.vglShape.getNFrames()
  
  def getBitsPerSample(self):
    return self.depth & 255
  
  """
    EQUIVALENT TO vglImage.h -> getWidthStep().
  """
  def getWidthStep(self):
    widthStep: int = None

    if( self.ipl is not None ):
      widthStep = vl.iplFindWidthStep(self.depth, self.getWidth() , self.nChannels)
    else:
      bps = self.getBitsPerSample()
      if( bps == 1 ):
        widthStep = (self.getWidthIn() - 1) / (8 + 1)
      elif( bps < 8 ):
        print("getWidthStep: Error: bits per pixel =", bps, " < 8 and != 1. Image depth may be wrong.")
        exit(1)
      else:
        widthStep = (bps / 8) * self.getNChannels() * self.getWidthIn()
    
    return widthStep
  
  def getTotalRows(self):
    return self.vglShape.getHeigthIn() * self.vglShape.getNFrames()
  
  def getTotalSizeInBytes(self):
    return self.getTotalRows() * self.getWidthStep()

"""
  EQUIVALENT TO iplImage.iplFindBitsPerSample(int depth)
"""
def iplFindBitsPerSample(depth):
  return depth & 255

"""
  EQUIVALENT TO iplImage.iplFindWidthStep(int depth, int width, int channels)
"""
def iplFindWidthStep(depth, width, channels=1):
  if( depth == vl.IPL_DEPTH_1U() ):
    return (width - 1) / (8 + 1)
  
  bpp = vl.iplFindBitsPerSample(depth)
  if(bpp < 8):
    print("iplFindWidthStep: Error: bits per pixel=", bpp, "and != 1. Image depth may be wrong.")
    exit(1)
  
  return (depth / 8) * channels * width

"""
  EQUIVALENT TO vglImage.vglImage3To4Channels()
"""
def vglImage3To4Channels(img):
  rgb_to_rgba(img)

"""
  EQUIVALENT TO vglImage.vglImage4To3Channels()
"""
def vglImage4To3Channels(img):
  rgba_to_rgb(img)

"""
  EQUIVALENT TO:
    vglImage.vglLoadImage
    vglImage.vglLoad3dImage
    vglImage.vglLoadNdImage
    vglImage.vglLoadPgm

  THIS METHOD READS THE IMAGE PATH AND OPEN
  IT AS A NUMPY NDARRAY. THROWS A ERROR MESSAGE
  IF THE PATH IS INCORRECT. BUILD THE MAKE THE CALL
  TO CONSTRUCT THE vglShape OBJECT. 
"""
def vglLoadImage(img, filename=""):
  if( img.filename == "" ):
    if( filename == "" ):
      print("vglImage: vglLoadImage Error: Image file path not defined! Empty string received!")
      exit()
    else:
      img.filename = filename
  try:
    img.ipl = io.imread(img.filename)
    vl.vglAddContext(img, vl.VGL_RAM_CONTEXT())
  except FileNotFoundError as fnf:
    print("vglImage: vglLoadImage Error: loading image from file:", img.filename)    
    print(str(fnf))
    exit()
  except Exception as e:
    print("vglImage: vglLoadImage Error: Unrecognized exception was thrown.")
    print(str(e))
    exit()
  
  if( isinstance(img.ipl, np.ndarray) ):
    #print("vglImage: Image loaded! VGL_RAM_CONTEXT.")

    vl.create_vglShape(img)

    img.depth = img.getVglShape().getNFrames()
    img.nChannels = img.getVglShape().getNChannels()
  
"""
  EQUIVALENT TO DIFFERENT IMAGE SAVE
  METHODS IN vglImage.cpp
"""
def vglSaveImage(filename, img):
  print("::Saving Picture in Hard Drive")
  io.imsave(filename, img.ipl)

"""
  THIS METHOD INITIATES THE vglShape OBJECT
  TO THE IMAGE. IT LOOKS IF IS A 2D OR A 3D IMAGE
  AND CONSTRUCT THE vglShape OBJECT.

  IN EVERY CHANGE IN THE IMAGE, THIS METHOD MUST BE
  CALLED, TO ASSURES THAT THE vglShape IS SYNCED WITH
  THE ACTUAL IMAGE.
"""
def create_vglShape(img):
  if(img.ipl is not None):
    #print("-> create_vglShape: Starting")

    img.vglShape = vl.VglShape()
    if( img.ndim == vl.VGL_IMAGE_2D_IMAGE() ):
      #print("2D Image")
      if( len(img.ipl.shape) == 2 ):
        # SHADES OF GRAY IMAGE
        #print("VglImage LUMINANCE")
        img.vglShape.constructor2DShape(1, img.ipl.shape[1], img.ipl.shape[0])
      elif(len(img.ipl.shape) == 3):
        # MORE THAN ONE COLOR CHANNEL
        #print("VglImage RGB")
        img.vglShape.constructor2DShape(img.ipl.shape[2], img.ipl.shape[1], img.ipl.shape[0])
    elif( img.ndim == vl.VGL_IMAGE_3D_IMAGE() ):
      #print("3D Image")
      if( len(img.ipl.shape) == 3 ):
        # SHADES OF GRAY IMAGE
        #print("VglImage LUMINANCE")
        img.vglShape.constructor3DShape( 1, img.ipl.shape[2], img.ipl.shape[1], img.ipl.shape[0] )
      elif(len(img.ipl.shape) == 4):
        # MORE THAN ONE COLOR CHANNEL
        #print("VglImage RGB")
        img.vglShape.constructor3DShape( img.ipl.shape[3], img.ipl.shape[2], img.ipl.shape[1], img.ipl.shape[0] )
    else:
      print("vglImage: create_vglShape Error: image dimension not recognized. img.ndim:", img.ndim)
      exit()
  else:
    print("vglImage: create_vglShape Error: img.ipl is None. Please, load image first!")
    exit()
  #print("<- create_vglShape: Ending\n")

"""
  EQUIVALENT TO vglImage.3To4Channels()
"""
def rgb_to_rgba(img):
  #print("::[RGB -> RGBA]")
  ipl_rgba = np.empty((img.vglShape.getHeigth(), img.vglShape.getWidth(), 4), img.ipl.dtype)

  ipl_rgba[:,:,0] = img.ipl[:,:,0]
  ipl_rgba[:,:,1] = img.ipl[:,:,1]
  ipl_rgba[:,:,2] = img.ipl[:,:,2]
  ipl_rgba[:,:,3] = 255

  img.ipl = ipl_rgba
  create_vglShape(img)

"""
  EQUIVALENT TO vglImage.3To4Channels()
"""
def rgba_to_rgb(img):
  #print("::[RGBA -> RGB]")
  if( (img.ipl[0,0,:].size < 4) | (img.ipl[0,0,:].size > 4) ):
    print("vglImage: rgba_to_rgb: Error: IMAGE IS NOT RGBA.")
    exit()
  else:
    ipl_rgb = np.empty((img.vglShape.getHeigth(), img.vglShape.getWidth(), 3), img.ipl.dtype)
    ipl_rgb[:,:,0] = img.ipl[:,:,0]
    ipl_rgb[:,:,1] = img.ipl[:,:,1]
    ipl_rgb[:,:,2] = img.ipl[:,:,2]

    img.ipl = ipl_rgb
    vl.create_vglShape(img)
