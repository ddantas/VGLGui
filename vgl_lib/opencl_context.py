import threading
import pyopencl as cl
import vgl_lib as vl
import sys, glob, os

"""
  THIS CLASS IS EQUIVALENT TO VglClContext
  STRUCT FOUNDED ON vglClImage.h

  HERE, IT IS USED JUST TO PASS THE PLATFORM ID,
  THE DEVICE ID, THE CONTEXT AND THE QUEUE OF THE
  DEVICE IN THE WAY IT IS FOUNDED ON C/C++ VERSION.

  commandQueue IS NOW A PROPERTY THAT RETURNS A PER-THREAD
  CommandQueue, ENABLING THREAD-SAFE PARALLEL EXECUTION.
"""
class VglClContext:
  def __init__(self, pl, dv, cn, ocl_ctx):
    self.platformId = pl
    self.deviceId   = dv
    self.context    = cn
    self._ocl_ctx   = ocl_ctx  # live reference to opencl_context

  @property
  def commandQueue(self):
    return self._ocl_ctx.queue

  @property
  def queue(self):
    return self.commandQueue

class opencl_context:
  """
    THIS CLASS MANAGES THE PyOpenCL INICIAL INSTANCIATION AND
    THE SYSTEM'S DEVICES AND ITS PROPERTIES (LIKE CONTEXT AND QUEUE).
    IT ALSO LOAD THE HEADERS AND CONSTANTS NEEDED TO COMPILE THE KERNELS.

    queue IS A THREAD-LOCAL PROPERTY: EACH THREAD GETS ITS OWN
    CommandQueue BACKED BY THE SHARED cl.Context.
  """

  def __init__(self, device_type):
    print("opencl_context: Instanciating Context and Queue...")
    self.platforms = cl.get_platforms()
    for p in self.platforms:
      self.devs = p.get_devices(device_type)
      if (len(self.devs) > 0):
        self.platform = p
        self.device = self.devs[0]
        break

    self.ctx = cl.Context([self.device])
    self._local = threading.local()  # per-thread storage for CommandQueue

    # PROGRAM VARIABLE. STORES ALL COMPILED KERNELS
    self.programs = []

  @property
  def queue(self):
    """Returns the CommandQueue for the current thread, creating it if needed."""
    if not hasattr(self._local, 'queue'):
      thread_name = threading.current_thread().name
      print(f"[opencl] nova CommandQueue para thread '{thread_name}'")
      self._local.queue = cl.CommandQueue(self.ctx)
    return self._local.queue

  @property
  def commandQueue(self):
    """Alias for queue — retrocompatibility."""
    return self.queue

  def finish_queue(self):
    """Waits for all operations in the current thread's queue to complete."""
    if hasattr(self._local, 'queue'):
      self._local.queue.finish()

  def is_kernel_compiled(self, method_name):
    for program in self.programs:
      if( method_name in program.kernel_names ):
        return program
    return None

  """
    THIS METHOD VERIFIES IF PATH TO KERNEL FILE EXISTS.
      IF EXISTS, IT VERIFIES IF THE KERNEL IS ALREADY COMPILED.
      IF IS NOT COMPILED, THE KERNEL IS BUILDED.
      IF THE KERNEL IS ALREADY COMPILED, IT DOES NOTHING.
  """
  def get_compiled_kernel(self, filepath, kernelname):
    #print("-> get_compiled_kernel: Starting")
    import os
    kernel_file = None

    try:
      kernel_file = open(filepath, "r")
    except FileNotFoundError as fnf:
      # IF FILE WASN'T NOT FOUND, PRINT THE PATH SENT AND THE CURRENT WORKING DIRECTORY.
      print("get_compiled_kernel: Error: Kernel File not found. Filepath:", filepath+", cwd:", os.getcwd())
      print(str(fnf))
      exit()
    except Exception as e:
      print("get_compiled_kernel: Error: A unexpected exception was thrown while trying to open kernel file. Filepath:", filepath)
      print(str(e))
      exit()

    program = self.is_kernel_compiled(kernelname)
    
    if( program is None ):
      #print("get_compiled_kernel: Building Kernel.")
      self.load_headers(filepath)
      program = cl.Program(self.ctx, kernel_file.read())
      self.programs.append( program.build(options=self.get_build_options()) )
      
    kernel_file.close()
    #print("<- get_compiled_kernel: Ending\n")
    return self.is_kernel_compiled(kernelname)
    
  """
    Making the vglClContext variables to retrocompatibility, where
      self.platformId = self.platform.int_ptr
      self.deviceId = self.device.int_ptr
      self.context = self.ctx
      self.commandQueue = self.queue
  """
  def get_vglClContext_attributes(self):
    return VglClContext(self.platform.int_ptr, self.device.int_ptr, self.ctx, self)
  
  def load_headers(self, filepath):
    #print("Loading Headers")
    self.kernel_file = open(filepath, "r")
    buildDir = self.getDir(filepath)

    self.build_options = "-I "+buildDir
    self.build_options = self.build_options + " -D VGL_SHAPE_NCHANNELS={0}".format(vl.VGL_SHAPE_NCHANNELS())
    self.build_options = self.build_options + " -D VGL_SHAPE_WIDTH={0}".format(vl.VGL_SHAPE_WIDTH())
    self.build_options = self.build_options + " -D VGL_SHAPE_HEIGTH={0}".format(vl.VGL_SHAPE_HEIGTH())
    self.build_options = self.build_options + " -D VGL_SHAPE_LENGTH={0}".format(vl.VGL_SHAPE_LENGTH())
    self.build_options = self.build_options + " -D VGL_MAX_DIM={0}".format(vl.VGL_MAX_DIM())
    self.build_options = self.build_options + " -D VGL_ARR_SHAPE_SIZE={0}".format(vl.VGL_ARR_SHAPE_SIZE())
    self.build_options = self.build_options + " -D VGL_ARR_CLSTREL_SIZE={0}".format(vl.VGL_ARR_CLSTREL_SIZE())
    self.build_options = self.build_options + " -D VGL_STREL_CUBE={0}".format(vl.VGL_STREL_CUBE())
    self.build_options = self.build_options + " -D VGL_STREL_CROSS={0}".format(vl.VGL_STREL_CROSS())
    self.build_options = self.build_options + " -D VGL_STREL_GAUSS={0}".format(vl.VGL_STREL_GAUSS())
    self.build_options = self.build_options + " -D VGL_STREL_MEAN={0}".format(vl.VGL_STREL_MEAN())

    # READING THE HEADER FILES BEFORE COMPILING THE KERNEL
    while( buildDir ):
      for file in glob.glob(buildDir+"/*.h"):
        #print(file)
        self.pgr = cl.Program(self.ctx, open(file, "r"))
      
      buildDir = self.getDir(buildDir)

  def getDir(self, filePath):
    size = len(filePath)-1
    bar = -1
    for i in range(0, size):
      if(filePath[i] == '/'):
        bar = i
        i = -1
    return filePath[:bar+1]
  
  # GETTERS
  def get_queue(self):
    return self.queue
  
  def get_context(self):
    return self.ctx
  
  def get_build_options(self):
    return self.build_options
