import multiprocessing as mp
import psutil as p
import numpy as np
from math import ceil
import scipy.sparse as sp
from scipy import fftpack as f

def process_func(shared_mem,nnz_count):
  name = mp.current_process().name
  i = name - 1

  local_mem = np.array(shared_mem[nnz_count*i:nnz_count*(i+2)]).view(complex)
  print "hello from worker {}\n".format(name), local_mem


#  print "hello world from worker {} with memory {} Mbs".format(name,
 #                                                          p.virtual_memory(
# ).free/1e6)

def slice_multiply(A, U, slice_index, shared_dict):
 shared_dict[slice_index] =  np.matmul(U.T, np.dot(A,U))

'''-----------------------------------------------------------------------------
    compute_fft(tensor_tubes)
        This function takes in a list of k (n x T) sparse matrices and computes 
      the fft transform for each row in each of sparse matrices, then puts 
      the results into a shared dense tensor of size (n x n x ceil(T/2)).
        Note that this is taking advantage of the symmetry of the fft applied 
      to real data. Also note that the input data coming in is from a rotated 
      n x n x T tensor, so the ijk element of the passed in tensor must be 
      mapped to the ikj spot in the dense tensor computed. 
    Input:
      tensor_tubes:
        a list of n x T sparse matrices to have the fft run on. 
      fft_P:
        a dense array of type double, which will hold the complex fourier 
        coefficients. Note that because the array is of type double, the real 
        and complex part must go in two adjacenct indices.
-----------------------------------------------------------------------------'''
def compute_fft(tensor_tubes,fft_P):

  core = mp.current_process().name - 1

  n = tensor_tubes[0].shape[0]
  T = tensor_tubes[0].shape[1]

  for j in xrange(len(tensor_tubes)):
    for i in xrange(n):
      #check if all zero
      if not tensor_tubes[j][i,:].nnz:
        fft = f.fft(tensor_tubes[j][i,:].todense())
        #b/c data is real elems 0, 1:ceil((T-1)/2) are unique
        for k in xrange(ceil((T-1)/2.0)):
          fft_P[n * (n * k + i) + core * j] = fft[k].real
          fft_P[n * (n * k + i) + core * j] = fft[k].imag


