from skimage import io
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
import vgl_lib as vl

"""
	ocl AND ocl_context ARE GLOBAL VARIABLES.
	ocl IS EQUIVALENT TO cl ON ORIGINAL vglClImage.
"""
ocl = None
ocl_context = None

"""
	struct_sizes IS A PYTHON-EXCLUSIVE GLOBAL VARIABLE.
	THIS VARIABLE CAN BE USED TO STORE THE WAY CURRENT
	OPENCL-DEVICE ORGANIZES THE vglClStrEl AND vglClShape
	DATA. WITH THE ORGANIZATION INFORMATION, THE BITWISE 
	PROCESS CAN BE DONE.
"""
struct_sizes = None

"""
	VARIABLE TO BINARY-IMAGES 
"""
bin_image_pack_size = None

"""
	EQUIVALENT TO vglClInit METHOD, FOUND ON
	vglClImage.vglClInit().

        default device_type is 2 (cl.device_type.CPU)
        alternative is         4 (cl.device_type.GPU)

"""
def vglClInit(device_type=2, ocl_context_a=None, ss_a=None, bin_image_pack_size_a=None):
	global ocl
	global ocl_context
	global struct_sizes
	global bin_image_pack_size
	
	# INITIATING OCL_CONTEXT
	if( ocl_context is None ):
		ocl_context = vl.opencl_context(device_type)
		ocl = ocl_context.get_vglClContext_attributes()
	else:
		ocl_context = ocl_context_a
		ocl = ocl_context.get_vglClContext_attributes()
	
	# INITIATING STRUCT SIZES
	if( ss_a is None ):
		ss = vl.struct_sizes()
		struct_sizes = ss.get_struct_sizes()
		ss = None
	else:
		struct_sizes = ss_a.get_struct_sizes()
	
	# INITIATING BIN_IMAGE_PACK_SIZE
	if((bin_image_pack_size_a == vl.PACK_SIZE_8())
	or (bin_image_pack_size_a == vl.PACK_SIZE_32())
	or (bin_image_pack_size_a == vl.PACK_SIZE_64()) ):
		bin_image_pack_size = bin_image_pack_size_a
	elif( bin_image_pack_size_a is None ):
		#print("vglClInit: Warning: Assuming bin_image_pack_size as 8")
		bin_image_pack_size = vl.PACK_SIZE_8()
	else:
		print("vglClInit: Error! bin_image_pack_size not 8, 32 or 64. Ending execution.")
		exit()

"""
	EQUIVALENT TO vglClImage.vglClUpload()
	TREATING ACCORDING TO clForceAsBuf.
"""
def vglClUpload(img):

	if( vl.vglIsInContext(img, vl.VGL_RAM_CONTEXT()) or vl.vglIsInContext(img, vl.VGL_BLANK_CONTEXT()) ):
		if( img.clForceAsBuf == vl.IMAGE_CL_OBJECT() ):
			vglClImageUpload(img)
		elif( img.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			vglClNdImageUpload(img)
		
		vl.vglAddContext(img, vl.VGL_CL_CONTEXT())
	else:
		print("vglClUpload: Error: image context is not in VGL_RAM_CONTEXT or VGL_BLANK_CONTEXT.")
		exit()

"""
	EQUIVALENT TO vglClImage.vglClDownload()
	TREATING ACCORDING TO clForceAsBuf.
"""
def vglClDownload(img):
	if( vl.vglIsInContext(img, vl.VGL_CL_CONTEXT()) ):
		if( img.clForceAsBuf == vl.IMAGE_CL_OBJECT() ):
			vglClImageDownload(img)
		elif( img.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			vglClNdImageDownload(img)

		vl.vglAddContext(img, vl.VGL_RAM_CONTEXT())
	else:
		print("vglClDownload: Error: image context is not in VGL_CL_CONTEXT.")
		exit()

"""
	IT TAKES THE RAM-SIDE IMAGE, ALLOCATES THE
	DEVICE-SIDE MEMORY, CONSTRUCTS THE IMAGE OBJECT
	AND UPLOADS THE IMAGE OBJECT TO THE DEVICE.
"""
def vglClImageUpload(img):
	global ocl
	mf = cl.mem_flags

	# IMAGE VARS
	#print("-> vglClImageUpload: Starting.")
	if( img.getVglShape().getNFrames() == 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeigth(), 1 )
		shape  = ( img.getVglShape().getWidth(), img.getVglShape().getHeigth() )

		imgFormat = cl.ImageFormat(vl.cl_channel_order(img), vl.cl_channel_type(img))
		img.oclPtr = cl.Image(ocl.context, mf.READ_WRITE, imgFormat, shape)
	elif( img.getVglShape().getNFrames() > 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeigth(), img.getVglShape().getNFrames() )
		shape = ( img.getVglShape().getWidth(), img.getVglShape().getHeigth(), img.getVglShape().getNFrames() )

		imgFormat = cl.ImageFormat(vl.cl_channel_order(img), vl.cl_channel_type(img))
		img.oclPtr = cl.Image(ocl.context, mf.READ_WRITE, imgFormat, shape)
	else:
		print("vglClImageUpload: VglImage NFrames wrong. NFrames returns:", img.getVglShape().getNFrames() )
		exit()

	# COPYING NDARRAY IMAGE TO OPENCL IMAGE OBJECT
	cl.enqueue_copy(ocl.commandQueue, img.get_oclPtr(), img.get_ipl(), origin=origin, region=region, is_blocking=True)
	#print("<- vglClImageUpload: Ending.\n")

"""
	THIS METHOD TAKES THE DEVICE-SIDE OPENCL IMAGE AND
	COPY IT BACK TO THE RAM-SIDE. IN THE END, THE METHOD
	CALL THE vglShape CONSTRUCTOR, TO ATUALIZE THE vglShape
	OBJECT.
"""
def vglClImageDownload(img):
	global ocl 

	# MAKE IMAGE DOWNLOAD HERE
	#print("-> vglClImageDownload: Starting")

	if( img.getVglShape().getNFrames() == 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeigth(), 1 )
		totalSize = img.getVglShape().getHeigth() * img.getVglShape().getWidth() * img.getVglShape().getNChannels()

		buffer = np.zeros(totalSize, img.get_ipl().dtype)
		cl.enqueue_copy(ocl.commandQueue, buffer, img.get_oclPtr(), origin=origin, region=region, is_blocking=True)

		if( img.getVglShape().getNChannels() == 1 ):
			buffer = np.frombuffer( buffer, img.get_ipl().dtype ).reshape( img.getVglShape().getHeigth(), img.getVglShape().getWidth() )
		elif( (img.getVglShape().getNChannels() == 3) or (img.getVglShape().getNChannels() == 4) ):
			buffer = np.frombuffer( buffer, img.get_ipl().dtype ).reshape( img.getVglShape().getHeigth(), img.getVglShape().getWidth(), img.getVglShape().getNChannels() )
	elif( img.getVglShape().getNFrames() > 1 ):
		#pitch = (0, 0)
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeigth(), img.getVglShape().getNFrames() )
		totalSize = img.getVglShape().getHeigth() * img.getVglShape().getWidth() * img.getVglShape().getNFrames()

		buffer = np.zeros(totalSize, img.get_ipl().dtype)
		cl.enqueue_copy(ocl.commandQueue, buffer, img.get_oclPtr(), origin=origin, region=region, is_blocking=True)


		if( img.getVglShape().getNChannels() == 1 ):
			buffer = np.frombuffer( buffer, img.get_ipl().dtype ).reshape( img.getVglShape().getNFrames(), img.getVglShape().getHeigth(), img.getVglShape().getWidth() )
		elif( (img.getVglShape().getNChannels() == 3) or (img.getVglShape().getNChannels() == 4) ):
			buffer = np.frombuffer( buffer, img.get_ipl().dtype ).reshape( img.getVglShape().getNFrames(), img.getVglShape().getHeigth(), img.getVglShape().getWidth(), img.getVglShape().getNChannels() )

	img.ipl = buffer
	#print("<- vglClImageDownload: Ending.\n")
	vl.create_vglShape(img)

"""
	IT TAKES THE RAM-SIDE IMAGE, ALLOCATES THE
	DEVICE-SIDE MEMORY, AND PASSES THE ND-ARRAY
	TO THE DEVICE.
"""
def vglClNdImageUpload(img):
	global ocl 
	mf = cl.mem_flags
	# CREATING DEVICE POINTER AND COPYING HOST TO DEVICE
	img.oclPtr = cl.Buffer(ocl.context, mf.READ_ONLY, img.get_ipl().nbytes)
	cl.enqueue_copy(ocl.commandQueue, img.get_oclPtr(), img.get_ipl().tobytes(), is_blocking=True)

"""
	THIS METHOD TAKES THE DEVICE-SIDE OPENCL BUFFER AND
	COPY IT BACK TO THE RAM-SIDE. THEN, THE METHOD CALLS
	THE vglShape CONSTRUCTOR, TO ATUALIZE THE vglShape
	OBJECT.
"""
def vglClNdImageDownload(img):
	global ocl
	cl.enqueue_copy(ocl.commandQueue, img.get_ipl(), img.get_oclPtr())
	vl.create_vglShape(img)

"""
	METHOD CURRENTLY NOT IN USE
	EQUIVALENT TO vglClImage.vglClCheckError
	THIS METHOD VERIFIES IF IS REALLY AN ERROR.
	IF IS, IT SHOWS THE ERROR ID AND PRINTS THE CAUSE. 
"""
def vglClCheckError(error, name):
	print("error:", error)
	print("name:", name)
	if(error < vl.CL_SUCCESS() and error >= vl.CL_MIN_ERROR()):
		print("Error", error, vl.vglClErrorMessages()[error], "while doing the following operation:")
		print(name)
		exit(error)

"""
	PYTHON-EXCLUSIVE METHODS
"""
# RETURNS the image_pack_size 
def get_bin_image_pack_size():
	global bin_image_pack_size
	return bin_image_pack_size

# RETURNS THE vl.VglClContext OBJECT
def get_ocl():
	return ocl

# RETURNS THE vl.opencl_context OBJECT
def get_ocl_context():
	return ocl_context

# SETS THE vl.VglClContext OBJECT
def set_ocl(ctx):
	global ocl
	if( isinstance(ctx, vl.VglClContext) ):
		ocl = ctx
	else:
		print("Error! not VglClContext object!")

# RETURNS struct_sizes ARRAY
def get_struct_sizes():
	global struct_sizes
	if( not isinstance(struct_sizes, np.ndarray) ):
		vl.vglClInit()
		print("get_struct_sizes: Warning! get_struct_sizes before vglClInit. Calling vglClInit now...")

	return struct_sizes

# RETURNS THE IMAGE CHANNEL TYPE
def cl_channel_type(img):
	oclPtr_dtype = None
	if( img.ipl.dtype == np.uint8 ):
		oclPtr_dtype = cl.channel_type.UNORM_INT8
		#print("cl_channel_type: 8bit Channel Size!")
	elif( img.ipl.dtype == np.uint16 ):
		oclPtr_dtype = cl.channel_type.UNORM_INT16
		#print("cl_channel_type: 16bit Channel Size!")

	return oclPtr_dtype

"""
	RETURNS THE IMAGE CHANNEL ORDER
	(OR QUANTITY OF CHANNELS)
"""
def cl_channel_order(img):
	oclPtr_channel_order = None
	if( img.getVglShape().getNChannels() == 1 ):
		oclPtr_channel_order = cl.channel_order.LUMINANCE
	elif( img.getVglShape().getNChannels() == 2 ):
		oclPtr_channel_order = cl.channel_order.RG
	elif( img.getVglShape().getNChannels() == 3 ):
		oclPtr_channel_order = cl.channel_order.RGB
	elif( img.getVglShape().getNChannels() == 4 ):
		oclPtr_channel_order = cl.channel_order.RGBA

	return oclPtr_channel_order

"""
	RETURNS A cl.Buffer or cl.Image OBJECT
	WITH THE SAME PROPERTIES OF THE img SENT
	AS ARGUMENT.
	IT IS USED TO CREATE A OUTPUT IMAGE WITH
	THE SAME SIZE AS THE INPUT IMAGE, FOR 
	EXAMPLE.
"""
def get_similar_oclPtr_object(img, nChannels=None):
	global ocl
	mf = cl.mem_flags

	channel_order = None
	if (nChannels is None):
		# print("ENTROU EM NONE")
		channel_order = vl.cl_channel_order(img)
	elif (nChannels == 1):
		print("ENTROU EM 1")
		channel_order = cl.channel_order.LUMINANCE
	elif(nChannels == 2):
		channel_order = cl.channel_order.RG
	elif(nChannels == 3):
		channel_order = cl.channel_order.RGB
	elif(nChannels == 4):
		channel_order = cl.channel_order.RGBA


	opencl_device = None
	if( isinstance(img.get_oclPtr(), cl.Image) ):
		#print("get_similar_oclPtr_object: oclPtr is cl.Image.")
		imgFormat = cl.ImageFormat(channel_order, vl.cl_channel_type(img))
		opencl_device = cl.Image(ocl.context, mf.READ_WRITE, imgFormat, img.get_oclPtr().shape )
	elif isinstance(img.get_oclPtr(), cl.Buffer):
		#print("get_similar_oclPtr_object: oclPtr is cl.Buffer.")
		opencl_device = cl.Buffer(ocl.context, mf.READ_WRITE, img.get_ipl().nbytes)

	return opencl_device

"""
	IT RETURNS A vl.Image OBJECT, WITH SIMILAR
	PROPERTIES AS THE img SENT AS ARGUMENT.
	img.ipl IS NOT SET, BUT OTHER PROPERTIES ARE.
	THE img.inContext IS SET AS VGL_BLANK_CONTEXT().
"""
def create_blank_image_as(img):
	image = vl.VglImage(img.filename, img.depth, img.ndim, img.clForceAsBuf)
	image.ipl	 = np.asarray(img.ipl, img.ipl.dtype)
	image.shape	 = img.shape
	image.vglShape	 = img.vglShape
	image.depth	 = img.depth
	image.nChannels	 = img.nChannels
	image.has_mipmap = img.has_mipmap
	image.inContext	 = vl.VGL_BLANK_CONTEXT()

	return image

"""
	IT RETURNS A vl.Image OBJECT, WITH SIMILAR
	PROPERTIES AS THE img SENT AS ARGUMENT.
	img.ipl IS NOT SET, BUT OTHER PROPERTIES ARE.
	THE img.inContext IS SET AS VGL_BLANK_CONTEXT().
"""
def create_blank_image_as_gray(img):
	image = vl.VglImage(img.filename, img.depth, img.ndim, img.clForceAsBuf)
	image.ipl	 = np.array((img.ipl.shape[0], img.ipl.shape[1]), img.ipl.dtype)
	image.shape	 = img.shape
	#image.vglShape	 = vl.vglShape.VglShape.vglCreateShape(s, img.vglShape.ndim)
	image.vglShape   = vl.VglShape()
	image.vglShape.constructor2DShape(1, img.ipl.shape[1], img.ipl.shape[0])
	
	image.depth	 = img.depth
	image.nChannels	 = 1
	image.has_mipmap = img.has_mipmap
	image.inContext	 = vl.VGL_BLANK_CONTEXT()

	image.set_oclPtr( vl.get_similar_oclPtr_object(img, 1) )
	
	return image

"""
	RETURNS A OPENCL BUFFER WITH DATA 
	OF VglClStrel TO KERNELS TO READ
"""
def get_vglstrel_opencl_buffer(strel):
	global ocl
	buf = cl.Buffer(ocl.context, cl.mem_flags.READ_ONLY, strel.nbytes)
	cl.enqueue_copy(ocl.commandQueue, buf, strel.tobytes(), is_blocking=True)
	return buf

"""
	RETURNS A OPENCL BUFFER WITH DATA 
	OF VglClShape TO KERNELS TO READ
"""
def get_vglshape_opencl_buffer(shape):
	global ocl
	buf = cl.Buffer(ocl.context, cl.mem_flags.READ_ONLY, shape.nbytes)
	cl.enqueue_copy(ocl.commandQueue, buf, shape.tobytes(), is_blocking=True)
	return buf
