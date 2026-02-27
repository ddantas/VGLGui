import vgl_lib as vl

def vglIsContextValid(x):
	return ( (x >= 1) and (x <= 15) )
	
def vglIsContextUnique(x):
	return ( (x == 0) or (x == 1) or (x == 2) or (x == 4) or (x == 8) or (x == 16) )

"""
	PYTHON LOGIC OPERATIONS PRECEDENCY WORKS SLIGHTELY DIFERENTLY
	FROM C++. TO ASSURE THAT PYTHON-VERSION OF vglIsInContext WILL 
	ALWAYS RETURN THE SAME RESULT AS CPP-VERSION, THE FUNCTION WAS
	REWRITED. IN testCode/test_vglIsInContext.c A TEST COMPARING THE
	EQUIVALENCY OF THE TWO VERSIONS WAS WRITED.

	THIS METHOD RECEIVES TWO ARGUMENTS:
		VglImage:	img
		Integer:	x
	IT VERIFY IF img IS IN THE CONTEXT ACCUSED BY x.
	x MUST BE ONE OF THE FOLLOWING CONSTANTS:
		VGL_BLANK_CONTEXT()
		VGL_RAM_CONTEXT()
		VGL_GL_CONTEXT() (NOT IMPLEMENTED IN PYTHON-SIDE YET)
		VGL_CUDA_CONTEXT() (NOT IMPLEMENTED IN PYTHON-SIDE YET)
		VGL_CL_CONTEXT()
"""
def vglIsInContext(img, x):
	img_context = img.inContext & x
	if( img_context > 0 ):
		return 1
	elif( (img.inContext is 0) and (x is 0) ):
		return 1
	else:
		return 0

"""
	EQUIVALENT TO vglContext.vglAddContext() METHOD.
	MUST CALL AFTER COPY. CONTEXT MUST BE UNIQUE.
	RETURN 0 IN CASE OF ERROR AND RESULTING CONTEXT IF SUCCESS
"""
def vglAddContext(img, context):
	if( not vglIsContextUnique(context) ):
		print("vglAddContext: Error: context =", context, "is not unique or invalid")
		exit()
	
	img.inContext = img.inContext | context
	return img.inContext

"""
	CALL AFTER PROCESSING OR CREATING IMAGE.
	CONTEXT MUST BE UNIQUE
	AFTER CREATING, USE VGL_BLANK_CONTEXT.
	RETURN 0 IN A ERROR OCCURS AND RESULTING CONTEXT IF SUCCESS.
"""
def vglSetContext(img, context):
	if( not vglIsContextUnique(context) and (context is not 0) ):
		print("vglSetContext: Error: context =", context, "is not unique")
		exit()
	
	img.inContext = context
	return img.inContext

"""
	CALL BEFORE USING IMAGE.
	context ARGUMENT MUST BE UNIQUE.
	RETURN 0 IF ERROR OCCURS AND RESULTING CONTEXT IF SUCCESS

	VGL_ERROR WAS CREATED IN ORDER TO DIFFERENTIATE
	WHEN vglCheckContext(img, context) RETURNED 0 (ERROR) 
	OR VGL_BLANK_CONTEXT THAT WAS ALSO 0. THEN, IN THE
	METHOD vglCheckContext(), WHEN context IS NOT UNIQUE
	OR A ERROR OCCURS, IS RETURNED VGL_ERROR.
	
	Shaders (vglCopy, vglDilateSq3, vglCudaInvertOnPlace etc)
	CALL ->
	vglCheckContext
	CALL ->
	vglUpload, vglDownload, vglGlToCuda, vglCudaToGl
	CALL ->
	vglSetContext, vglAddContext
"""
def vglCheckContext(img, context):
	if( not vglIsContextUnique( context ) ):
		print("vglCheckContext: Error: context =", context, "is not unique or invalid")
		exit()
	
	if( vglIsInContext(img, context) ):
		#print("vglCheckContext: image already in context", context)
		return context
	"""
		HERE STARTS THE CASE-LIKE SEQUENCE FROM 
		vglContext.vglCheckContext(VglImage img, int context)
	"""
	# IF THE CONTEXT IS IN RAM
	if( context is vl.VGL_RAM_CONTEXT() ):
		# AND IS A BLANK IMAGE, IT IS ON RAM. JUST SET IT IN CONTEXT.
		if( vglIsInContext(img, vl.VGL_BLANK_CONTEXT() ) ):
			vglAddContext(img, vl.VGL_RAM_CONTEXT())
		# AND THE CONTEXT IS IN CL-DEVICE, DOWNLOAD IT BACK TO RAM.
		if( vglIsInContext(img, vl.VGL_CL_CONTEXT() ) ):
			vl.vglClDownload(img)
		# CV context images are already numpy arrays in RAM
		if( vglIsInContext(img, vl.VGL_CV_CONTEXT() ) ):
			vglAddContext(img, vl.VGL_RAM_CONTEXT())
	# IF THE CONTEXT IS IN CL-DEVICE
	elif( context is vl.VGL_CL_CONTEXT() ):
		# AND IS A BLANK-IMAGE, IT IS ON RAM. UPLOAD IT TO CL-DEVICE!
		if( vglIsInContext( img, vl.VGL_BLANK_CONTEXT() ) ):
			vl.vglClUpload(img)
		# AND THE CONTEXT IS IN RAM, UPLOAD IT TO CL-DEVICE!
		if( vglIsInContext( img, vl.VGL_RAM_CONTEXT() ) ):
			vl.vglClUpload(img)
		# From CV to CL: CV data is in RAM (img.ipl), just upload
		if( vglIsInContext( img, vl.VGL_CV_CONTEXT() ) ):
			vl.vglClUpload(img)
	# IF THE CONTEXT IS IN CV (OpenCV/CPU)
	elif( context == vl.VGL_CV_CONTEXT() ):
		# CV operates on img.ipl (numpy array), same as RAM
		if( vglIsInContext(img, vl.VGL_BLANK_CONTEXT() ) ):
			vglAddContext(img, vl.VGL_CV_CONTEXT())
		if( vglIsInContext(img, vl.VGL_RAM_CONTEXT() ) ):
			vglAddContext(img, vl.VGL_CV_CONTEXT())
		if( vglIsInContext(img, vl.VGL_CL_CONTEXT() ) ):
			vl.vglClDownload(img)
			vglAddContext(img, vl.VGL_CV_CONTEXT())
	else:
		print("vglCheckContext: Error: Trying to copy to invalid context =", context)
		exit()
	
	return img.inContext

"""
	CALL BEFORE WRITING TO IMAGE. CONTEXT MUST BE UNIQUE.

    GL AND RAM IMAGES ARE ALWAYS ALLOCATED. 
	CUDA IMAGES MUST BE ALLOCATED IF cudaPbo == -1.

	BY NOW, ALL CUDA-SPECIFIC CODE WILL BE IGNORED 
	ON THIS METHOD. FUTURE INCREMENTS CAN IMPROVE THAT.
"""
def vglCheckContextForOutput(img, context):
	if( not (img is None) ):
		print("img is not null")
		return 1
		
	return 0
