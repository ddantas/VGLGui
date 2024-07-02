import numpy as np 
import vgl_lib as vl

"""
	EQUIVALENT TO vglClStrEl struct
	LOCATED IN vglClStrEl.h
"""
class VglClStrEl(object):
	def __init__(self, ndim=0, size=0):
		self.data = np.zeros((vl.VGL_ARR_CLSTREL_SIZE()), np.float32)
		self.ndim = np.int32(0)
		self.shape = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.offset = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.size = np.int32(0)

"""
	EQUIVALENT TO VglStrEl OBJECT
	LOCATED IN vglStrEl.cpp
"""
class VglStrEl(object):
	def __init__(self):
		self.vglShape = vl.VglShape()
		self.data = np.zeros((1), np.float32)

	"""
		EQUIVALENT TO vglStrEl.VglCreateStrEl()
		RECEIVES TWO PARAMETERS:
			data: ARRAY WITH CONVOLUTION WINDOWS ELEMENTS.
			vglShape: vglShape OBJECT TO HELP MANIPULATE THE Structuring Element.
	"""
	def VglCreateStrEl(self, data, vglShape):
		size = vglShape.getSize()
		self.vglShape = vl.VglShape()
		self.vglShape.constructorFromVglShape(vglShape)
		self.data = np.zeros((size), np.float32)

		for i in range(0, size):
			self.data[i] = data[i]

	"""
		EQUIVALENT TO vglStrEl.VglCreateStrEl()
		RECEIVES TWO PARAMETERS:
			data: array with convolution window elements.
			vglShape: shape, or dimension sizes, associated with array.
	"""
	def constructorFromDataVglShape(self, data, vglShape):
		self.VglCreateStrEl(data, vglShape)

	"""
		EQUIVALENT TO THE vglStrEl CONSTRUCTOR
		THAT RECEIVES TWO PARAMETERS:
			type: (VGL_SHAPE_CUBE|VGL_SHAPE_CROSS|VGL_SHAPE_GAUSS)
			ndim: number of dimensions.
		
		ATTENTION:
			In the .cpp file, is used the name "type"
			but "type" is a reserved name in Python.
			Then, on this object, the name "Type" will replace "type".
	"""
	def constructorFromTypeNdim(self, Type, ndim):
		shape = np.zeros(vl.VGL_MAX_DIM(), np.int32)
		shape[0] = 1
		shape[2] = 1

		for i in range(1, ndim+1): # ndim+1 cuz in c++ is for i <= ndim
			shape[i] = 3

		vglShape = vl.VglShape()
		vglShape.constructorFromShapeNdimBps(shape, ndim)

		size = vglShape.getSize()
		data = np.zeros((size), np.float32)
		index = 0

		if( Type == vl.VGL_STREL_CROSS() ):
			coord = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)

			for i in range(0, size):
				data[i] = np.float32(0.0)

			for d in range(1, ndim+1):
				coord[d] = np.int32(1)

			index = vglShape.getIndexFromCoord(coord)
			data[index] = np.float32(1.0)

			for d in range(1, ndim+1):
				coord[d] = np.int32(0)
				index = vglShape.getIndexFromCoord(coord)
				data[index] = np.float32(1.0)

				coord[d] = np.int32(2)
				index = vglShape.getIndexFromCoord(coord)
				data[index] = np.float32(1.0)

				coord[d] = 1

		elif( Type == vl.VGL_STREL_GAUSS() ):
			coord = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)
			coord[0] = np.int32(0)
			size = vglShape.getSize()

			for i in range(0, size):
				val = np.float32(1.0)
				vglShape.getCoordFromIndex(i, coord)

				for d in range(1, ndim+1):
					if( coord[d] == 1 ):
						val = val * np.float32(0.5)
					else:
						val = val * np.float32(0.25)

				data[i] = val

		elif( Type == vl.VGL_STREL_MEAN() ):
			for i in range(0, size):
				data[i] = 1.0 / size

		elif( Type == vl.VGL_STREL_CUBE() ):
			for i in range(0, size):
				data[i] = 1.0
		else:
			for i in range(0, size):
				data[i] = 1.0
			
		#print(data)
		#print(size)
		self.constructorFromDataVglShape(data, vglShape)

	"""
		DEFAULT GETTERS
	"""
	def getData(self):
		return self.data

	def getSize(self):
		return self.vglShape.getSize()

	def getNpixels(self):
		return self.vglShape.getNpixels()

	def getNdim(self):
		return self.vglShape.getNdim()

	def getShape(self):
		return self.vglShape.getShape()

	def getOffset(self):
		return self.vglShape.getOffset()

	"""
		EQUIVALENT TO THE vglStrEl.asVglStrEl() METHOD
	"""
	def asVglClStrEl(self):
		result = VglClStrEl()
		shape = self.vglShape.asVglClShape()

		size = self.getSize()
		if( size > vl.VGL_ARR_CLSTREL_SIZE() ):
			print("vglStrEl: asVglClStrEl Error: structuring element size > VGL_ARR_CLSTREL_SIZE. Change this value in vglClStrEl.h to a greater one.")
			exit()

		result.ndim = np.int32(self.vglShape.getNdim())
		result.size = np.int32(self.vglShape.getSize())
		
		for i in range(0, vl.VGL_MAX_DIM()+1):
			result.shape[i] = np.int32(shape.shape[i])
			result.offset[i] = np.int32(shape.offset[i])
		
		for i in range(0, size):
			result.data[i] = np.int32(self.data[i])

		return result
	
	"""
		ON C/C++ VERSION, asVglClStrEl WOULD BE ENOUGH.

		PYTHON-SIDE MUST TREAT THIS DATA BEFORE RETURN IT.
		HERE FOLLOWS THOSE TREATMENTS 
	"""
	def get_asVglClStrEl_buffer(self):
		struct_sizes = vl.get_struct_sizes()
		image_cl_strel = self.asVglClStrEl()

		strel_obj = np.zeros(struct_sizes[0], np.uint8)
		
		self.copy_into_byte_array(image_cl_strel.data,	strel_obj, struct_sizes[1])
		self.copy_into_byte_array(image_cl_strel.shape,	strel_obj, struct_sizes[2])
		self.copy_into_byte_array(image_cl_strel.offset,strel_obj, struct_sizes[3])
		self.copy_into_byte_array(image_cl_strel.ndim,	strel_obj, struct_sizes[4])
		self.copy_into_byte_array(image_cl_strel.size,	strel_obj, struct_sizes[5])

		return vl.get_vglstrel_opencl_buffer(strel_obj)

	"""
		PYTHON-ONLY METHODS
	"""
	def copy_into_byte_array(self, value, byte_array, offset):
		for iterator, byte in enumerate( value.tobytes() ):
			byte_array[iterator+offset] = byte