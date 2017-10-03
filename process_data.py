import scipy.sparse as sp
from scipy.sparse.linalg import svds, LinearOperator
from sys import argv
import matplotlib.pylab as plt
from math import log
import numpy as np
import cProfile
from functools import reduce
import csv
#from memory_profiler import profile

#Global Variables
DATA_FILE_PATH = "/mnt/hgfs/datasets/wordEmbeddings/"
FILE_NAME = DATA_FILE_PATH + "wordPairPMI_2016.csv"
INDEX_WORD_FILE = DATA_FILE_PATH + "wordIDHash_min200.csv"
SING_VAL_EXTENSION = "SingVals.npy"

UPDATE_FREQUENCY_CONSTANT = 10.0

#argv[1].split(".")[0] +

#run by global filelocation or argument if passed in
def main():
  #profile_read_in_function()
  pmi = read_in_pmi(FILE_NAME,True) \
 #   if (len(argv) < 2) else read_in_pmi(argv[1],True)

def test_func():
  f = open(FILE_NAME,"r")
  f.next()

  edges = {}
  edge_count = 0
  for line in f:
     edges[edge_count] = line.split(',')
     edge_count += 1

  f.close()

'''-----------------------------------------------------------------------------
    read_in_pmi(filename, display_progress)
      This function takes in a filename and returns a pmi matrix stored in 
      the location. The file is assumed to be formatted as 
        line ->  word_index, context_index, pmi_value
      The file then builds a scipy dok_matrix and returns it.
    Input:
      filename - (string) 
        location of the file.
      return_scaled_count - (optional bool)
        instead returning the pmi for a word context pair, we can return the 
        number of times that the pairs appears together in a text corpus, 
        this provides naive weights for the loss function.  
      display_progress - (optional bool) 
        display statements updating the progress of the file load or not.
    Returns:
      pmi - (dok_matrix)
        a sparse matrix with the corresponding pmi values for each of the 
        word context pairs. 
    NOTE: read_in_word_index current returns the word count over all text 
    documents, not the word count for an individual text corpus, rendering 
    the return_scaled_count option invalid
-----------------------------------------------------------------------------'''
def read_in_pmi(filename, return_scaled_count = False,
                display_progress = False):
  f = open(filename,"r")
  f.next() # skip word, context, pmi line
  total_edge_count = 0
  clean_indices = 0
  edges = {}
  i_max = -1
  j_max = -1

  word_indices = read_in_word_index(return_scaled_count)


  '''
  the count is over all words over all times
  #compute word count TODO: VALIDATE SUM
  if return_scaled_count:
    word_count = reduce(lambda x,key: x + word_indices[key][1],word_indices)
  '''
  if display_progress:
    print 'Read in word indices'
  new_indices = {}

  #count the edges in the file, dimensions of PPMI matrix, and reassign indices.
  for line in f:
    edge = line.split(',')

    #reassign new indices to prevent empty submatrices in pmi
    word_ID = int(edge[0])
    context_ID = int(edge[1])

    word = word_indices[word_ID] if not return_scaled_count else\
           word_indices[word_ID][0]
    context = word_indices[context_ID] if not return_scaled_count else\
              word_indices[context_ID][0]


    if word not in new_indices:
      new_indices[word] = clean_indices
      clean_indices += 1

    if context not in new_indices:
      new_indices[context] = clean_indices
      clean_indices += 1

    edge_val = np.float(edge[2])
    if return_scaled_count:
      edge_val =  np.exp(edge_val)* \
                  word_indices[word_ID][1] * word_indices[context_ID][1]

    edges[total_edge_count] = [new_indices[word], new_indices[context],
                               edge_val]
    #check if new indices are largest row or column found
    if new_indices[word] > i_max:
      i_max = new_indices[word]

    if new_indices[context] > j_max:
      j_max = new_indices[context]

    total_edge_count += 1

  f.close()

  if display_progress:
    print "counted {} edges over {} by {} words"\
      .format(total_edge_count, i_max, j_max)


  # initialize counts for updating user as file loads
  if display_progress:
    update_frequency = total_edge_count / UPDATE_FREQUENCY_CONSTANT
    edge_count = 0

  shape = (i_max+1,j_max+1)
  #initialize sparse matrix
  pmi = sp.dok_matrix(shape)

  for i in xrange(total_edge_count):
    print i
    print edges[i][0], edges[i][1], edges[i][2]
    pmi[edges[i][0], edges[i][1]] = edges[i][2]
    if display_progress:
      edge_count += 1
      if edge_count > 100:
        break
      if edge_count % update_frequency == 0:
        print "{}% complete, {} edges read in"\
          .format((edge_count/float(total_edge_count))*100,
                  edge_count)

  return pmi

'''
  f.close()
  f = open(filename, "r")
  f.next()

  #initialize counts for updating user as file loads
  if display_progress:
    update_frequency = total_edge_count/UPDATE_FREQUENCY_CONSTANT
    edge_count = 0

  shape = (i_max+1,j_max+1)
  #initialize sparse matrix
  pmi = sp.dok_matrix(shape)

  #reiterate through to store non-zeros
  for line in f:
    edge = line.split(',')
    i = new_indices[word_indices[int(edge[0])]]  #arrays are indexed by 0
    j = new_indices[word_indices[int(edge[1])]]

    pmi[i, j] = np.float(edge[2])
    if display_progress:
      edge_count += 1
      if edge_count > 100:
        break
      if edge_count % update_frequency == 0:
        print "{}% complete, {} edges read in"\
          .format((edge_count/total_edge_count)*100,
                  edge_count)
'''
def profile_read_in_function():
  #cProfile.run('test_func()')
  cProfile.run('read_in_pmi(FILE_NAME,True)')

'''-----------------------------------------------------------------------------
    read_in_word_index()
      This function reads in the word index associated with the text corpus, 
      from the wordIDHash_min200.csv file so the embeddings can be associated 
      with the proper word rather than an integer ID. 
    Inputs:
      include_word_count - (boolean)
        a boolean indicating whether or not the word count should be included in
        as the second element in a tuple in the key value pair. 
    Returns:
      word_IDs - (dictionary)
        a dictionary which has indices as the keys and the words as the 
        associated values. If include_word_count is true, then the values are 
        2-tuples where the first element is the word, and the 2nd element is 
        the word frequency. 
-----------------------------------------------------------------------------'''
def read_in_word_index(include_word_count = False):
  f = open(INDEX_WORD_FILE, "r") # formatted as wordId, word, word count
  word_IDs = {}
  for line in f:
    word_stat = line.split(',')
    if include_word_count:
      word_IDs[int(word_stat[0])] = (word_stat[1], int(word_stat[2]))
    else:
      word_IDs[int(word_stat[0])] = word_stat[1]

  return word_IDs

'''-----------------------------------------------------------------------------
    convert_pmi_to_window_count()
      This function takes in a pmi matrix and returns a matrix where the (w,c)th
      element of |D| element corpus is equal to  
        (# of times word w appears in context c)*|D|
      This is used to create a weighted objective function such that the (w,c)th 
      softmax term is weighted by the number of times that context appears in 
      the corpus(including the scaling factor |D|).
    Input:
      pmi (n x m) sparse matrix
        the pmi matrix to convert
-----------------------------------------------------------------------------'''

'''-----------------------------------------------------------------------------
    matrix_stats(matrix)
      This function takes in a sparse matrix and returns a collection of 
      statistics about the matrix in question. data reported about the matrix 
      includes
        ROWS, COLUMNS, NON-ZEROS,and SINGULAR_VALUES
    Input:
      matrix - (n x m sparse matrix)
        the matrix in question to report matrix stats about
    Returns:
      stats - (dictionary)
        a dictionary of the stats to be reported back in the where the keys 
        are the listed matrix stats reported above. 
-----------------------------------------------------------------------------'''
def matrix_stats(matrix):
  stats = {}
  stats["ROWS"] = matrix.shape[0]
  stats["COLS"] = matrix.shape[1]
  stats["SINGULAR_VALUES"] = svds(mat_vec(matrix, 3), k=10,
                                  return_singular_vectors=False)
  return stats

'''-----------------------------------------------------------------------------
    matrix_visualization(matrix)
      This function takes in a matrix and uses plt functions to visualize the 
      non-zeros of the matrix. 
    Input:
      matrix - (n x m sparse matrix)
        the matrix to visualize.
-----------------------------------------------------------------------------'''
def matrix_visualization(matrix):
  plt.spy(matrix)
  plt.show()

'''-----------------------------------------------------------------------------
    mat_vec(matrix, vector)
       This function produces an anonymous function to be used as a linear 
       operator in the scipy svd routine.
    Input:
      matrix - (n x m sparse matrix)
        The pmi matrix to use to compute the word embeddings. 
      k - (int)
        The negative sample multiple factor.
    Returns:
      mat_vec - (m-vec -> n-vec)
        an anonymous function which works as an O(m) linear operator which 
        adds a rank 1 update to the pmi matrix.   (M - log(k))
    Notes:
      Unclear if the numpy sum function has numerical instability issues. 
-----------------------------------------------------------------------------'''
def mat_vec(matrix, k):
  logFactor = log(k)
  n = matrix.shape[0]
  m = matrix.shape[1]
  mat_vec = lambda v: (matrix * v) + (np.ones(n) * v.sum() * logFactor)
  rmat_vec = lambda v: (matrix.T * v) + (np.ones(m) * v.sum() * logFactor)
  return LinearOperator((n,m), mat_vec, rmatvec = rmat_vec)



if __name__ == "__main__":
 main()
