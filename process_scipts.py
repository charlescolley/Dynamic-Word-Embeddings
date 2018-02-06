import multiprocessing as mp
import psutil as p
import numpy as np
from math import floor
import scipy.sparse as sp
from scipy import fftpack as f
import word2vec as w2v

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
    compute_svd(A, filename_prefix,d)
        This function is just a wrapper for computing a collection of SVDs 
        for a list of sparse matrices and then saving the results in parallel. 
    Input:
      A - (n x m sparse matrix)
        a sparse matix 
      filename_prefix - (string)
        the place to store the singular vectors and values
      d - (integer)
        the number of singular pairs to compute
      linear_operator - (optional string)
        a string which indicates which type of linear operator should be made 
        from the matrix before running the svd routine
        Options:
          mean_center - 
            a linear operator which mean centers any vector being mutiplied to 
            the matrix, and the output of the linear mapping.
-----------------------------------------------------------------------------'''
def compute_svd(A,filename_prefix,d, linear_operator = None):
  if linear_operator:
      u, s, _ = sp.linalg.svds(w2v.mean_center(A), k = d, return_singular_vectors
      = 'u')
      filename_prefix = filename_prefix + '_' + linear_operator+ '_'
  else:
    u, s, _ = sp.linalg.svds(A, k=d, return_singular_vectors = 'u')
  np.save(filename_prefix + 'U',u)
  np.save(filename_prefix + 'sigma',s)

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
        a list of n x T sparse dok matrices run fft on. 
      fft_P:
        a dense array of type double, which will hold the complex fourier 
        coefficients. Note that because the array is of type double, the real 
        and complex part must go in two adjacent indices.
-----------------------------------------------------------------------------'''
def compute_fft(tensor_tubes,fft_P,col_offset):

  core = mp.current_process().name - 1

  n = tensor_tubes[0].shape[0]
  T = 1 + int(floor((tensor_tubes[0].shape[1]/2)))

  for j in xrange(len(tensor_tubes)):
    for i in xrange(n):
      #check if all zero
      if tensor_tubes[j][i,:].nnz:
        fft = f.fft(tensor_tubes[j][i,:].todense())[0]
        #b/c data is real elems 0, 1:ceil((T-1)/2) are unique
        for k in xrange(T):
          #factor of 2 accounts for additional mode-3 elements for the complex
          # components
          fft_P[2*T * (n * i + (col_offset + j)) + 2*k] = fft[k].real
          fft_P[2*T * (n * i + (col_offset + j)) + 2*k+1] = fft[k].imag


