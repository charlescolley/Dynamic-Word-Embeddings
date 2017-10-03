import theano.tensor as T
import tensorflow as tf
import theano as t
import downhill
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import process_data as pd

def main():
  pmi = pd.read_in_pmi()
  svd_embedding(pd.filter_up_to_kth_largest(pmi,10000))

'''
   svd_embedding(pmi, k):
     compute a 3-dimensional embedding for a given pmi matrix and plot it 
     using a 3-D scatter plot. 
'''
def svd_embedding(pmi, k):
  if k == 0:
    U,s = svds(pmi, k=3)
  else:
    U,s = svds(mat_vec(matrix, k), k=3)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection ='3d')
  # .text(x, y, z, s, zdir=None, **kwargs
  ax.scatter(xs = U[:,0],ys = U[:,1],zs = U[:,2])
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
  return LinearOperator((n, m), mat_vec, rmatvec=rmat_vec)

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
'''
    build_objective_functions(word_count_matrix, k)
      This function takes in a n x m matrix with the scaled number of times a 
      word appears within the context c (#(w,c)) and returns a lambda 
      function which computes the loss function and a gradient function. The 
      function will 
    Inputs:
      word_count_matrix - (sparse n x m matrix)
        a scipy sparse matrix which the (i,j)th entry corresponds to #(w_i,
        c_j)|D|. Here |D| denotes the number of words in the text corpus. 
      word_count - (dictionary)
        a dictionary which has the word counts for a text corpus. 
      k - int
        the negative sampling rate, creates k fake samples which help prevent 
        unform distriutions from arising. Samples are created from a unigram 
        distriution. P((w,c)) = #(w)*#(c)^{3/4}/Z where Z is a normalizing 
        constant. 
    Returns:
      loss_func - (lambda func)
        an anonymous function which has the negated word2vec objective 
        function (which is typically maximized). 
        \sum_{(w,c) \in D} (log(softmax(v_c,v_w)) - k 
      gradient - (lambda func)
        an anonymous function which has the gradient of the 
'''
def build_loss_function(word_count_matrix, word_count, k):
  print TODO

def tensorflow_tutorial():
  vocab_size = 10000
  embedding_size = 100
  embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1,1))


def theano_code():
  x = t.shared(np.ones(2),'x')
  y = t.shared(np.ones(2), 'y')

  matrix = np.array([[1,2],[3,4]])

  x_1, x_2 = T.dvectors('x_1','x_2')
  A = T.dmatrix('A')
  qf = T.dot(x,T.dot(A,y))
  sigma = 1 / (1 + T.exp(-y))
  logistic = t.function([y],sigma)

  print downhill.minimize(qf, matrix,inputs=A)


def downhill_example():

#  THEANO_FLAGS = None
  m = t.shared(np.ones((1, ), dtype=np.float64), name='m')
  b = t.shared(np.zeros((1, ), dtype=np.float64), name='b')

  x = T.vector('x')
  y = T.vector('y')

  loss = T.sqr(m * x + b - y).sum()
  sizes = np.array([1200,2013,8129,2431,2211])
  prices = np.array([103020, 203310, 3922013, 224321, 449020])

  downhill.adaptive.ADAGRAD(loss, params=[sizes,prices],inputs=[x,y])
  #downhill.minimize(loss,[sizes, prices],inputs = [x,y])

  print m, b


#minimize a quadratic form


#def word2vec_loss_func(pmi):




if __name__ == "__main__":
  main()