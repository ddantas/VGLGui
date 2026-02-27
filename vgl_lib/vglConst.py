"""
	AS PYTHON DOESN'T HAVE CONSTANT DECLARATION, THE NEXT METHODS
	RETURN THE VALUES WHO NEED CONSTANT BEHAVIOR.
"""

# VGL_WIN DEFINES [pack64 branch]
def VGL_WIN_X0():
	return -1

def VGL_WIN_X1():
	return 1

def VGL_WIN_DX():
	return VGL_WIN_X1() - VGL_WIN_X0()

def VGL_WIN_Y0():
	return -1

def VGL_WIN_Y1():
	return 1

def VGL_WIN_DY():
	return VGL_WIN_Y1() - VGL_WIN_Y0()

def VGL_MIN_WINDOW_SPLIT():
	return 1

def VGL_DEFAULT_WINDOW_SPLIT():
	return 2

def VGL_MAX_WINDOW_SPLIT():
	return 4

def VGL_MAX_WINDOWS():
	return VGL_MAX_WINDOW_SPLIT() * VGL_MAX_WINDOW_SPLIT()

# VGL OPENCL CONSTANTS (TO getWidthStep())
def IPL_DEPTH_1U():
	return 1

# VGL CONSTANTS
def VGL_SHAPE_NCHANNELS():
	return VGL_SHAPE_D0()

def VGL_SHAPE_WIDTH():
	return VGL_SHAPE_D1()

def VGL_SHAPE_HEIGTH():
	return VGL_SHAPE_D2()

def VGL_SHAPE_LENGTH():
	return VGL_SHAPE_D3()

def VGL_4D():
	return 3

def VGL_MAX_DIM():
	return 10

def VGL_ARR_SHAPE_SIZE():
	return VGL_MAX_DIM()+1

def VGL_SHAPE_D0():
	return 0

def VGL_SHAPE_D1():
	return 1

def VGL_SHAPE_D2():
	return 2

def VGL_SHAPE_D3():
	return 3

def VGL_SHAPE_D4():
	return 4

def VGL_ARR_CLSTREL_SIZE():
	return 256

# vglStrEl WINDOW MODES
def VGL_STREL_CUBE():
	return 1

def VGL_STREL_CROSS():
	return 2

def VGL_STREL_GAUSS():
	return 3

def VGL_STREL_MEAN():
	return 4

# IMAGE DIMENSIONS CONSTANTS
def VGL_IMAGE_3D_IMAGE():
	return 3

def VGL_IMAGE_2D_IMAGE():
	return 2

# CONTEXT CONSTANTS
"""
	VGL_ERROR WAS CREATED IN ORDER TO DIFFERENTIATE
	WHEN vglCheckContext(img, context) RETURNED 0 (ERROR) 
	OR VGL_BLANK_CONTEXT THAT WAS ALSO 0. THEN, IN THE
	METHOD vglCheckContext(), WHEN context IS NOT UNIQUE
	OR A ERROR OCCURS, IS RETURNED VGL_ERROR.
"""
def VGL_BLANK_CONTEXT():
	return 0

def VGL_RAM_CONTEXT():
	return 1

def VGL_GL_CONTEXT():
	return 2

def VGL_CUDA_CONTEXT():
	return 4

def VGL_CL_CONTEXT():
	return 8

def VGL_CV_CONTEXT():
	return 16

def VGL_ERROR():
	return -1

# PYTHON-SPECIFIC IMAGE MANIPULATION MODE
def IMAGE_CL_OBJECT():
	return 0

def IMAGE_ND_ARRAY():
	return 1

# ERROR MESSAGES
def vglClErrorMessages():
	return [ 
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "CL_MISALIGNED_SUB_BUFFER_OFFSET",
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
    "CL_COMPILE_PROGRAM_FAILURE",
    "CL_LINKER_NOT_AVAILABLE",
    "CL_LINK_PROGRAM_FAILURE",
    "CL_DEVICE_PARTITION_FAILED",
    "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
    "UNKNOWN",
    "UNKNOWN",
    "UNKNOWN",
    "UNKNOWN",
    "UNKNOWN",
    "UNKNOWN",
    "UNKNOWN",
    "UNKNOWN",
    "UNKNOWN",
    "UNKNOWN",
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE",
    "CL_INVALID_PROPERTY",
    "CL_INVALID_IMAGE_DESCRIPTOR",
    "CL_INVALID_COMPILER_OPTIONS",
    "CL_INVALID_LINKER_OPTIONS",
    "CL_INVALID_DEVICE_PARTITION_COUNT"]

def CL_SUCCESS():
	return 0

def CL_MIN_ERROR():
	return -68

# PACK64 CONSTANTS
import pyopencl as cl
import numpy as np
import vgl_lib as vl

def VGL_PACK_SIZE_BITS_8():
	return 8

def VGL_PACK_SIZE_BYTES_8():
	return 1

def VGL_PACK_MAX_UINT_8():
	return 0xff

def VGL_PACK_CL_CONST_TYPE_8():
	return cl.channel_type.UNSIGNED_INT8

def VGL_PACK_CL_SHADER_TYPE_8():
	return np.uint8

def VGL_PACK_OUTPUT_SWAP_MASK_8():
	outputSwapMask = np.ndarray(32, VGL_PACK_CL_SHADER_TYPE_8)
	outputSwapMask[0] = VGL_PACK_CL_SHADER_TYPE_8()(0x80)
	outputSwapMask[1] = VGL_PACK_CL_SHADER_TYPE_8()(0x40)
	outputSwapMask[2] = VGL_PACK_CL_SHADER_TYPE_8()(0x20)
	outputSwapMask[3] = VGL_PACK_CL_SHADER_TYPE_8()(0x10)
	outputSwapMask[4] = VGL_PACK_CL_SHADER_TYPE_8()(0x08)
	outputSwapMask[5] = VGL_PACK_CL_SHADER_TYPE_8()(0x04)
	outputSwapMask[6] = VGL_PACK_CL_SHADER_TYPE_8()(0x02)
	outputSwapMask[7] = VGL_PACK_CL_SHADER_TYPE_8()(0x01)
	return outputSwapMask

def VGL_PACK_OUTPUT_DIRECT_MASK_8():
	outputDirectMask = np.ndarray(32, VGL_PACK_CL_SHADER_TYPE_8)
	outputDirectMask[0] = VGL_PACK_CL_SHADER_TYPE_8()(0x80)
	outputDirectMask[1] = VGL_PACK_CL_SHADER_TYPE_8()(0x40)
	outputDirectMask[2] = VGL_PACK_CL_SHADER_TYPE_8()(0x20)
	outputDirectMask[3] = VGL_PACK_CL_SHADER_TYPE_8()(0x10)
	outputDirectMask[4] = VGL_PACK_CL_SHADER_TYPE_8()(0x08)
	outputDirectMask[5] = VGL_PACK_CL_SHADER_TYPE_8()(0x04)
	outputDirectMask[6] = VGL_PACK_CL_SHADER_TYPE_8()(0x02)
	outputDirectMask[7] = VGL_PACK_CL_SHADER_TYPE_8()(0x01)
	return outputDirectMask

def VGL_PACK_SIZE_BITS_32():
	return 32

def VGL_PACK_SIZE_BYTES_32():
	return 4

def VGL_PACK_MAX_UINT_32():
	return 0xffffffff

def VGL_PACK_CL_CONST_TYPE_32():
	return cl.channel_type.UNSIGNED_INT32

def VGL_PACK_CL_SHADER_TYPE_32():
	return np.uint32

def VGL_PACK_OUTPUT_SWAP_MASK_32():
	outputSwapMask = np.ndarray(32, VGL_PACK_CL_SHADER_TYPE_32)
	outputSwapMask[0] = VGL_PACK_CL_SHADER_TYPE_32()(0x80)
	outputSwapMask[1] = VGL_PACK_CL_SHADER_TYPE_32()(0x40)
	outputSwapMask[2] = VGL_PACK_CL_SHADER_TYPE_32()(0x20)
	outputSwapMask[3] = VGL_PACK_CL_SHADER_TYPE_32()(0x10)
	outputSwapMask[4] = VGL_PACK_CL_SHADER_TYPE_32()(0x08)
	outputSwapMask[5] = VGL_PACK_CL_SHADER_TYPE_32()(0x04)
	outputSwapMask[6] = VGL_PACK_CL_SHADER_TYPE_32()(0x02)
	outputSwapMask[7] = VGL_PACK_CL_SHADER_TYPE_32()(0x01)
	outputSwapMask[8] = VGL_PACK_CL_SHADER_TYPE_32()(0x8000)
	outputSwapMask[9] = VGL_PACK_CL_SHADER_TYPE_32()(0x4000)
	outputSwapMask[10] = VGL_PACK_CL_SHADER_TYPE_32()(0x2000)
	outputSwapMask[11] = VGL_PACK_CL_SHADER_TYPE_32()(0x1000)
	outputSwapMask[12] = VGL_PACK_CL_SHADER_TYPE_32()(0x0800)
	outputSwapMask[13] = VGL_PACK_CL_SHADER_TYPE_32()(0x0400)
	outputSwapMask[14] = VGL_PACK_CL_SHADER_TYPE_32()(0x0200)
	outputSwapMask[15] = VGL_PACK_CL_SHADER_TYPE_32()(0x0100)
	outputSwapMask[16] = VGL_PACK_CL_SHADER_TYPE_32()(0x800000)
	outputSwapMask[17] = VGL_PACK_CL_SHADER_TYPE_32()(0x400000)
	outputSwapMask[18] = VGL_PACK_CL_SHADER_TYPE_32()(0x200000)
	outputSwapMask[19] = VGL_PACK_CL_SHADER_TYPE_32()(0x100000)
	outputSwapMask[20] = VGL_PACK_CL_SHADER_TYPE_32()(0x080000)
	outputSwapMask[21] = VGL_PACK_CL_SHADER_TYPE_32()(0x040000)
	outputSwapMask[22] = VGL_PACK_CL_SHADER_TYPE_32()(0x020000)
	outputSwapMask[23] = VGL_PACK_CL_SHADER_TYPE_32()(0x010000)
	outputSwapMask[24] = VGL_PACK_CL_SHADER_TYPE_32()(0x80000000)
	outputSwapMask[25] = VGL_PACK_CL_SHADER_TYPE_32()(0x40000000)
	outputSwapMask[26] = VGL_PACK_CL_SHADER_TYPE_32()(0x20000000)
	outputSwapMask[27] = VGL_PACK_CL_SHADER_TYPE_32()(0x10000000)
	outputSwapMask[28] = VGL_PACK_CL_SHADER_TYPE_32()(0x08000000)
	outputSwapMask[29] = VGL_PACK_CL_SHADER_TYPE_32()(0x04000000)
	outputSwapMask[30] = VGL_PACK_CL_SHADER_TYPE_32()(0x02000000)
	outputSwapMask[31] = VGL_PACK_CL_SHADER_TYPE_32()(0x01000000)
	return outputSwapMask

def VGL_PACK_OUTPUT_DIRECT_MASK_32():
	outputDirectMask = np.ndarray(64, VGL_PACK_CL_SHADER_TYPE_32)
	outputDirectMask[0] = VGL_PACK_CL_SHADER_TYPE_32()(0x80)
	outputDirectMask[1] = VGL_PACK_CL_SHADER_TYPE_32()(0x40)
	outputDirectMask[2] = VGL_PACK_CL_SHADER_TYPE_32()(0x20)
	outputDirectMask[3] = VGL_PACK_CL_SHADER_TYPE_32()(0x10)
	outputDirectMask[4] = VGL_PACK_CL_SHADER_TYPE_32()(0x08)
	outputDirectMask[5] = VGL_PACK_CL_SHADER_TYPE_32()(0x04)
	outputDirectMask[6] = VGL_PACK_CL_SHADER_TYPE_32()(0x02)
	outputDirectMask[7] = VGL_PACK_CL_SHADER_TYPE_32()(0x01)
	outputDirectMask[8] = VGL_PACK_CL_SHADER_TYPE_32()(0x8000)
	outputDirectMask[9] = VGL_PACK_CL_SHADER_TYPE_32()(0x4000)
	outputDirectMask[10] = VGL_PACK_CL_SHADER_TYPE_32()(0x2000)
	outputDirectMask[11] = VGL_PACK_CL_SHADER_TYPE_32()(0x1000)
	outputDirectMask[12] = VGL_PACK_CL_SHADER_TYPE_32()(0x0800)
	outputDirectMask[13] = VGL_PACK_CL_SHADER_TYPE_32()(0x0400)
	outputDirectMask[14] = VGL_PACK_CL_SHADER_TYPE_32()(0x0200)
	outputDirectMask[15] = VGL_PACK_CL_SHADER_TYPE_32()(0x0100)
	outputDirectMask[16] = VGL_PACK_CL_SHADER_TYPE_32()(0x800000)
	outputDirectMask[17] = VGL_PACK_CL_SHADER_TYPE_32()(0x400000)
	outputDirectMask[18] = VGL_PACK_CL_SHADER_TYPE_32()(0x200000)
	outputDirectMask[19] = VGL_PACK_CL_SHADER_TYPE_32()(0x100000)
	outputDirectMask[20] = VGL_PACK_CL_SHADER_TYPE_32()(0x080000)
	outputDirectMask[21] = VGL_PACK_CL_SHADER_TYPE_32()(0x040000)
	outputDirectMask[22] = VGL_PACK_CL_SHADER_TYPE_32()(0x020000)
	outputDirectMask[23] = VGL_PACK_CL_SHADER_TYPE_32()(0x010000)
	outputDirectMask[24] = VGL_PACK_CL_SHADER_TYPE_32()(0x80000000)
	outputDirectMask[25] = VGL_PACK_CL_SHADER_TYPE_32()(0x40000000)
	outputDirectMask[26] = VGL_PACK_CL_SHADER_TYPE_32()(0x20000000)
	outputDirectMask[27] = VGL_PACK_CL_SHADER_TYPE_32()(0x10000000)
	outputDirectMask[28] = VGL_PACK_CL_SHADER_TYPE_32()(0x08000000)
	outputDirectMask[29] = VGL_PACK_CL_SHADER_TYPE_32()(0x04000000)
	outputDirectMask[30] = VGL_PACK_CL_SHADER_TYPE_32()(0x02000000)
	outputDirectMask[31] = VGL_PACK_CL_SHADER_TYPE_32()(0x01000000)
	return outputDirectMask

def VGL_PACK_SIZE_BITS_64():
	return 64

def VGL_PACK_SIZE_BYTES_64():
	return 8

def VGL_PACK_MAX_UINT_64():
	return 0xffffffffffffffff

def VGL_PACK_CL_CONST_TYPE_64():
	return cl.channel_type.UNSIGNED_INT32

def VGL_PACK_CL_SHADER_TYPE_64():
	return np.uint64

def VGL_PACK_OUTPUT_SWAP_MASK_64():
	outputSwapMask = np.ndarray(64, VGL_PACK_CL_SHADER_TYPE_64)
	outputSwapMask[0] = VGL_PACK_CL_SHADER_TYPE_64()(0x80)
	outputSwapMask[1] = VGL_PACK_CL_SHADER_TYPE_64()(0x40)
	outputSwapMask[2] = VGL_PACK_CL_SHADER_TYPE_64()(0x20)
	outputSwapMask[3] = VGL_PACK_CL_SHADER_TYPE_64()(0x10)
	outputSwapMask[4] = VGL_PACK_CL_SHADER_TYPE_64()(0x08)
	outputSwapMask[5] = VGL_PACK_CL_SHADER_TYPE_64()(0x04)
	outputSwapMask[6] = VGL_PACK_CL_SHADER_TYPE_64()(0x02)
	outputSwapMask[7] = VGL_PACK_CL_SHADER_TYPE_64()(0x01)
	outputSwapMask[8] = VGL_PACK_CL_SHADER_TYPE_64()(0x8000)
	outputSwapMask[9] = VGL_PACK_CL_SHADER_TYPE_64()(0x4000)
	outputSwapMask[10] = VGL_PACK_CL_SHADER_TYPE_64()(0x2000)
	outputSwapMask[11] = VGL_PACK_CL_SHADER_TYPE_64()(0x1000)
	outputSwapMask[12] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800)
	outputSwapMask[13] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400)
	outputSwapMask[14] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200)
	outputSwapMask[15] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100)
	outputSwapMask[16] = VGL_PACK_CL_SHADER_TYPE_64()(0x800000)
	outputSwapMask[17] = VGL_PACK_CL_SHADER_TYPE_64()(0x400000)
	outputSwapMask[18] = VGL_PACK_CL_SHADER_TYPE_64()(0x200000)
	outputSwapMask[19] = VGL_PACK_CL_SHADER_TYPE_64()(0x100000)
	outputSwapMask[20] = VGL_PACK_CL_SHADER_TYPE_64()(0x080000)
	outputSwapMask[21] = VGL_PACK_CL_SHADER_TYPE_64()(0x040000)
	outputSwapMask[22] = VGL_PACK_CL_SHADER_TYPE_64()(0x020000)
	outputSwapMask[23] = VGL_PACK_CL_SHADER_TYPE_64()(0x010000)
	outputSwapMask[24] = VGL_PACK_CL_SHADER_TYPE_64()(0x80000000)
	outputSwapMask[25] = VGL_PACK_CL_SHADER_TYPE_64()(0x40000000)
	outputSwapMask[26] = VGL_PACK_CL_SHADER_TYPE_64()(0x20000000)
	outputSwapMask[27] = VGL_PACK_CL_SHADER_TYPE_64()(0x10000000)
	outputSwapMask[28] = VGL_PACK_CL_SHADER_TYPE_64()(0x08000000)
	outputSwapMask[29] = VGL_PACK_CL_SHADER_TYPE_64()(0x04000000)
	outputSwapMask[30] = VGL_PACK_CL_SHADER_TYPE_64()(0x02000000)
	outputSwapMask[31] = VGL_PACK_CL_SHADER_TYPE_64()(0x01000000)
	outputSwapMask[32] = VGL_PACK_CL_SHADER_TYPE_64()(0x8000000000)
	outputSwapMask[33] = VGL_PACK_CL_SHADER_TYPE_64()(0x4000000000)
	outputSwapMask[34] = VGL_PACK_CL_SHADER_TYPE_64()(0x2000000000)
	outputSwapMask[35] = VGL_PACK_CL_SHADER_TYPE_64()(0x1000000000)
	outputSwapMask[36] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800000000)
	outputSwapMask[37] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400000000)
	outputSwapMask[38] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200000000)
	outputSwapMask[39] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100000000)
	outputSwapMask[40] = VGL_PACK_CL_SHADER_TYPE_64()(0x800000000000)
	outputSwapMask[41] = VGL_PACK_CL_SHADER_TYPE_64()(0x400000000000)
	outputSwapMask[42] = VGL_PACK_CL_SHADER_TYPE_64()(0x200000000000)
	outputSwapMask[43] = VGL_PACK_CL_SHADER_TYPE_64()(0x100000000000)
	outputSwapMask[44] = VGL_PACK_CL_SHADER_TYPE_64()(0x080000000000)
	outputSwapMask[45] = VGL_PACK_CL_SHADER_TYPE_64()(0x040000000000)
	outputSwapMask[46] = VGL_PACK_CL_SHADER_TYPE_64()(0x020000000000)
	outputSwapMask[47] = VGL_PACK_CL_SHADER_TYPE_64()(0x010000000000)
	outputSwapMask[48] = VGL_PACK_CL_SHADER_TYPE_64()(0x80000000000000)
	outputSwapMask[49] = VGL_PACK_CL_SHADER_TYPE_64()(0x40000000000000)
	outputSwapMask[50] = VGL_PACK_CL_SHADER_TYPE_64()(0x20000000000000)
	outputSwapMask[51] = VGL_PACK_CL_SHADER_TYPE_64()(0x10000000000000)
	outputSwapMask[52] = VGL_PACK_CL_SHADER_TYPE_64()(0x08000000000000)
	outputSwapMask[53] = VGL_PACK_CL_SHADER_TYPE_64()(0x04000000000000)
	outputSwapMask[54] = VGL_PACK_CL_SHADER_TYPE_64()(0x02000000000000)
	outputSwapMask[55] = VGL_PACK_CL_SHADER_TYPE_64()(0x01000000000000)
	outputSwapMask[56] = VGL_PACK_CL_SHADER_TYPE_64()(0x8000000000000000)
	outputSwapMask[57] = VGL_PACK_CL_SHADER_TYPE_64()(0x4000000000000000)
	outputSwapMask[58] = VGL_PACK_CL_SHADER_TYPE_64()(0x2000000000000000)
	outputSwapMask[59] = VGL_PACK_CL_SHADER_TYPE_64()(0x1000000000000000)
	outputSwapMask[60] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800000000000000)
	outputSwapMask[61] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400000000000000)
	outputSwapMask[62] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200000000000000)
	outputSwapMask[63] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100000000000000)
	return outputSwapMask

def VGL_PACK_OUTPUT_DIRECT_MASK_64():
	outputDirectMask = np.ndarray(64, VGL_PACK_CL_SHADER_TYPE_64)
	outputDirectMask[0] = VGL_PACK_CL_SHADER_TYPE_64()(0x01)
	outputDirectMask[1] = VGL_PACK_CL_SHADER_TYPE_64()(0x02)
	outputDirectMask[2] = VGL_PACK_CL_SHADER_TYPE_64()(0x04)
	outputDirectMask[3] = VGL_PACK_CL_SHADER_TYPE_64()(0x08)
	outputDirectMask[4] = VGL_PACK_CL_SHADER_TYPE_64()(0x10)
	outputDirectMask[5] = VGL_PACK_CL_SHADER_TYPE_64()(0x20)
	outputDirectMask[6] = VGL_PACK_CL_SHADER_TYPE_64()(0x40)
	outputDirectMask[7] = VGL_PACK_CL_SHADER_TYPE_64()(0x80)
	outputDirectMask[8] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100)
	outputDirectMask[9] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200)
	outputDirectMask[10] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400)
	outputDirectMask[11] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800)
	outputDirectMask[12] = VGL_PACK_CL_SHADER_TYPE_64()(0x1000)
	outputDirectMask[13] = VGL_PACK_CL_SHADER_TYPE_64()(0x2000)
	outputDirectMask[14] = VGL_PACK_CL_SHADER_TYPE_64()(0x4000)
	outputDirectMask[15] = VGL_PACK_CL_SHADER_TYPE_64()(0x8000)
	outputDirectMask[16] = VGL_PACK_CL_SHADER_TYPE_64()(0x010000)
	outputDirectMask[17] = VGL_PACK_CL_SHADER_TYPE_64()(0x020000)
	outputDirectMask[18] = VGL_PACK_CL_SHADER_TYPE_64()(0x040000)
	outputDirectMask[19] = VGL_PACK_CL_SHADER_TYPE_64()(0x080000)
	outputDirectMask[20] = VGL_PACK_CL_SHADER_TYPE_64()(0x100000)
	outputDirectMask[21] = VGL_PACK_CL_SHADER_TYPE_64()(0x200000)
	outputDirectMask[22] = VGL_PACK_CL_SHADER_TYPE_64()(0x400000)
	outputDirectMask[23] = VGL_PACK_CL_SHADER_TYPE_64()(0x800000)
	outputDirectMask[24] = VGL_PACK_CL_SHADER_TYPE_64()(0x01000000)
	outputDirectMask[25] = VGL_PACK_CL_SHADER_TYPE_64()(0x02000000)
	outputDirectMask[26] = VGL_PACK_CL_SHADER_TYPE_64()(0x04000000)
	outputDirectMask[27] = VGL_PACK_CL_SHADER_TYPE_64()(0x08000000)
	outputDirectMask[28] = VGL_PACK_CL_SHADER_TYPE_64()(0x10000000)
	outputDirectMask[29] = VGL_PACK_CL_SHADER_TYPE_64()(0x20000000)
	outputDirectMask[30] = VGL_PACK_CL_SHADER_TYPE_64()(0x40000000)
	outputDirectMask[31] = VGL_PACK_CL_SHADER_TYPE_64()(0x80000000)
	outputDirectMask[32] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100000000)
	outputDirectMask[33] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200000000)
	outputDirectMask[34] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400000000)
	outputDirectMask[35] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800000000)
	outputDirectMask[36] = VGL_PACK_CL_SHADER_TYPE_64()(0x1000000000)
	outputDirectMask[37] = VGL_PACK_CL_SHADER_TYPE_64()(0x2000000000)
	outputDirectMask[38] = VGL_PACK_CL_SHADER_TYPE_64()(0x4000000000)
	outputDirectMask[39] = VGL_PACK_CL_SHADER_TYPE_64()(0x8000000000)
	outputDirectMask[40] = VGL_PACK_CL_SHADER_TYPE_64()(0x010000000000)
	outputDirectMask[41] = VGL_PACK_CL_SHADER_TYPE_64()(0x020000000000)
	outputDirectMask[42] = VGL_PACK_CL_SHADER_TYPE_64()(0x040000000000)
	outputDirectMask[43] = VGL_PACK_CL_SHADER_TYPE_64()(0x080000000000)
	outputDirectMask[44] = VGL_PACK_CL_SHADER_TYPE_64()(0x100000000000)
	outputDirectMask[45] = VGL_PACK_CL_SHADER_TYPE_64()(0x200000000000)
	outputDirectMask[46] = VGL_PACK_CL_SHADER_TYPE_64()(0x400000000000)
	outputDirectMask[47] = VGL_PACK_CL_SHADER_TYPE_64()(0x800000000000)
	outputDirectMask[48] = VGL_PACK_CL_SHADER_TYPE_64()(0x01000000000000)
	outputDirectMask[49] = VGL_PACK_CL_SHADER_TYPE_64()(0x02000000000000)
	outputDirectMask[50] = VGL_PACK_CL_SHADER_TYPE_64()(0x04000000000000)
	outputDirectMask[51] = VGL_PACK_CL_SHADER_TYPE_64()(0x08000000000000)
	outputDirectMask[52] = VGL_PACK_CL_SHADER_TYPE_64()(0x10000000000000)
	outputDirectMask[53] = VGL_PACK_CL_SHADER_TYPE_64()(0x20000000000000)
	outputDirectMask[54] = VGL_PACK_CL_SHADER_TYPE_64()(0x40000000000000)
	outputDirectMask[55] = VGL_PACK_CL_SHADER_TYPE_64()(0x80000000000000)
	outputDirectMask[56] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100000000000000)
	outputDirectMask[57] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200000000000000)
	outputDirectMask[58] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400000000000000)
	outputDirectMask[59] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800000000000000)
	outputDirectMask[60] = VGL_PACK_CL_SHADER_TYPE_64()(0x1000000000000000)
	outputDirectMask[61] = VGL_PACK_CL_SHADER_TYPE_64()(0x2000000000000000)
	outputDirectMask[62] = VGL_PACK_CL_SHADER_TYPE_64()(0x4000000000000000)
	outputDirectMask[63] = VGL_PACK_CL_SHADER_TYPE_64()(0x8000000000000000)
	return outputDirectMask

def PACK_SIZE_8():
	return 8

def PACK_SIZE_32():
	return 32

def PACK_SIZE_64():
	return 64

def VGL_PACK_SIZE_BITS():
	if( vl.get_bin_image_pack_size is None ):
		vl.vglClInit()

	if( vl.get_bin_image_pack_size == vl.PACK_SIZE_8() ):
		return VGL_PACK_SIZE_BITS_8()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_32() ):
		return VGL_PACK_SIZE_BITS_32()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_64() ):
		return VGL_PACK_SIZE_BITS_64()

	print("VGL_PACK_SIZE_BITS: Error! get_bin_image_pack_size not 8, 32 or 64.")
	exit()

def VGL_PACK_SIZE_BYTES():
	if( vl.get_bin_image_pack_size is None ):
		vl.vglClInit()

	if( vl.get_bin_image_pack_size == vl.PACK_SIZE_8() ):
		return VGL_PACK_SIZE_BYTES_8()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_32() ):
		return VGL_PACK_SIZE_BYTES_32()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_64() ):
		return VGL_PACK_SIZE_BYTES_64()

	print("VGL_PACK_SIZE_BYTES: Error! get_bin_image_pack_size not 8, 32 or 64.")
	exit()

def VGL_PACK_MAX_UINT():
	if( vl.get_bin_image_pack_size is None ):
		vl.vglClInit()

	if( vl.get_bin_image_pack_size == vl.PACK_SIZE_8() ):
		return VGL_PACK_MAX_UINT_8()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_32() ):
		return VGL_PACK_MAX_UINT_32()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_64() ):
		return VGL_PACK_MAX_UINT_64()

	print("VGL_PACK_MAX_UINT: Error! get_bin_image_pack_size not 8, 32 or 64.")
	exit()

def VGL_PACK_CL_CONST_TYPE():
	if( vl.get_bin_image_pack_size is None ):
		vl.vglClInit()

	if( vl.get_bin_image_pack_size == vl.PACK_SIZE_8() ):
		return VGL_PACK_CL_CONST_TYPE_8()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_32() ):
		return VGL_PACK_CL_CONST_TYPE_32()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_64() ):
		return VGL_PACK_CL_CONST_TYPE_64()

	print("VGL_PACK_CL_CONST_TYPE: Error! get_bin_image_pack_size not 8, 32 or 64.")
	exit()

def VGL_PACK_CL_SHADER_TYPE():
	if( vl.get_bin_image_pack_size is None ):
		vl.vglClInit()

	if( vl.get_bin_image_pack_size == vl.PACK_SIZE_8() ):
		return VGL_PACK_CL_SHADER_TYPE_8()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_32() ):
		return VGL_PACK_CL_SHADER_TYPE_32()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_64() ):
		return VGL_PACK_CL_SHADER_TYPE_64()

	print("VGL_PACK_CL_SHADER_TYPE: Error! get_bin_image_pack_size not 8, 32 or 64.")
	exit()

def VGL_PACK_OUTPUT_SWAP_MASK():
	if( vl.get_bin_image_pack_size is None ):
		vl.vglClInit()

	if( vl.get_bin_image_pack_size == vl.PACK_SIZE_8() ):
		return VGL_PACK_OUTPUT_SWAP_MASK_8()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_32() ):
		return VGL_PACK_OUTPUT_SWAP_MASK_32()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_64() ):
		return VGL_PACK_OUTPUT_SWAP_MASK_64()

	print("VGL_PACK_OUTPUT_SWAP_MASK: Error! get_bin_image_pack_size not 8, 32 or 64.")
	exit()

def VGL_PACK_OUTPUT_DIRECT_MASK():
	if( vl.get_bin_image_pack_size is None ):
		vl.vglClInit()

	if( vl.get_bin_image_pack_size == vl.PACK_SIZE_8() ):
		return VGL_PACK_OUTPUT_DIRECT_MASK_8()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_32() ):
		return VGL_PACK_OUTPUT_DIRECT_MASK_32()
	elif( vl.get_bin_image_pack_size == vl.PACK_SIZE_64() ):
		return VGL_PACK_OUTPUT_DIRECT_MASK_64()

	print("VGL_PACK_SIZE_BITS: Error! get_bin_image_pack_size not 8, 32 or 64.")
	exit()