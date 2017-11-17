import multiprocessing as mp
import psutil as p
from scipy import fftpack as f

def process_func():
  name = mp.current_process().name
  print "hello world from worker {} with memory {} Mbs".format(name,
                                                           p.virtual_memory().free/1e6)


'''-----------------------------------------------------------------------------
    compute_fft(tensor_tubes)
        This function takes in a list of k (n x T) sparse matrices and computes 
      the fft transform for each row in each of sparse matrices, then puts 
      the results into a shared dense tensor of size (n x n x ceil(T/2)). 
      Note that this is taking advantage of the symmetry of the fft applied 
      to real data. Also note that the input data coming in is from a rotated 
      n x n x T tensor, so the ijk element of the passed in tensor must be 
      mapped to the ikj spot in the dense tensor computed. 
-----------------------------------------------------------------------------'''
def compute_fft(tensor_tubes):
