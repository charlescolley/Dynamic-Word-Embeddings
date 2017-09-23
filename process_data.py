import scipy.sparse
import numpy as np

#Global Variables
DATA_FILE_PATH = "/mnt/hgfs/datasets/wordEmbeddings/"
FILE_NAME = DATA_FILE_PATH + "wordPairPMI_2016.csv"
UPDATE_FREQUENCY_CONSTANT = 13

def main():
  read_in_pmi(FILE_NAME,True)


'''-----------------------------------------------------------------------------
    read_in_pmi(filename, display_progress)
      This function takes in a filename and returns a pmi matrix stored in 
      the location. The file is assumed to be formatted as 
        line ->  word_index, context_index, pmi_value
      The file then builds a scipy dok_matrix and returns it.
    Input:
      filename - (string) 
        location of the file.
      display_progress - (optional bool) 
        display statements updating the progress of the file load or not.
    Returns:
      pmi - (dok_matrix)
        a sparse matrix with the corresponding pmi values for each of the 
        word context pairs. 
 -----------------------------------------------------------------------------'''
def read_in_pmi(filename, display_progress = False):
  f = open(filename,"r")
  f.next() # skip word, context, pmi line

  edge_count = 0
  i_max = -1
  j_max = -1
  #count the edges in the file, and dimensions of PPMI matrix
  for line in f:
    edge_count += 1
    edge = line.split(',')
    i = int(edge[0])
    j = int(edge[1])
    if (i > i_max):
      i_max = i
    if (j > j_max):
      j_max = j

  if display_progress:
    print "counted {} edges over {} by {} words"\
      .format(edge_count, i_max, j_max)

  f.close()
  f = open(filename, "r")
  f.next()

  #initialize counts for updating user as file loads
  if display_progress:
    update_frequency = edge_count/UPDATE_FREQUENCY_CONSTANT
    edge_count = 0

  shape = (max(i_max, j_max), max(i_max, j_max))
  #initialize sparse matrix
  pmi = scipy.sparse.dok_matrix(shape)

  #reiterate through to store non-zeros
  for line in f:
    edge = line.split(',')
    i = int(edge[0]) - 1  #arrays are indexed by 0
    j = int(edge[1]) - 1
    pmi[i, j] = float(edge[2])
    if display_progress:
      edge_count += 1
      if edge_count % update_frequency == 0:
        print "{}% complete, {} edges read in"\
          .format((edge_count/update_frequency) * UPDATE_FREQUENCY_CONSTANT,
                  edge_count)

  return pmi


'''-----------------------------------------------------------------------------
    matrix_stats(matrix)
      This function takes in a sparse matrix and returns a collection of 
      statistics about the matrix in question. data reported about the matrix 
      includes
        ROWS, COLUMNS, NON-ZEROS,and SINGULAR VALUES
    Input:
      matrix - (n x m sparse matrix)
        the matrix in question to report matrix stats about
    Returns:
      stats - (dictionary)
        a dictionary of the stats to be reported back in the where the keys 
        are the listed matrix stats reported above. 
-----------------------------------------------------------------------------'''
def matrix_stats(matrix):


if __name__ == "__main__":
 main()
