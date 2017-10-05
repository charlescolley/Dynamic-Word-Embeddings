import theano.tensor as T
import tensorflow as tf
import theano as t
import downhill
import numpy as np
from math import log
import scipy.sparse as sp
from scipy.sparse.linalg import svds, LinearOperator
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import process_data as pd

def main():
  pmi = pd.read_in_pmi()
 # pmi = sp.rand(100,100,format='dok')
  svd_embedding(pd.filter_up_to_kth_largest(pmi,100000),0)

'''
   svd_embedding(pmi, k):
     compute a 3-dimensional embedding for a given pmi matrix and plot it 
     using a 3-D scatter plot. 
'''
def svd_embedding(matrix, k):
  if k == 0:
    results = svds(matrix, k=3)
  else:
    results = svds(mat_vec(matrix, k), k=3)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection ='3d')
  # .text(x, y, z, s, zdir=None, **kwargs
  ax.scatter(xs = results[0][:,0],ys = results[0][:,1],zs = results[0][:,2])
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

'''-----------------------------------------------------------------------------
   svd_grad_U(P,U,V)
     This function returns the gradient of the function 
     
     .5*\|P - UV^T\|_F^2 +
        \frac{\lambda_1}{2}\|U\|_F^2 + \frac{\lambda_1}{2}\|U\|_F^2
        
     which evaluates to (P + I\lambda_2)V - U(I\lambda + V^TV). The function 
     will output a dense (n x d) matrix where n is the number of words in the 
     vocabulary and d is the dimension of the word embedding.
   Inputs:
     P - (n x n sparse matrix)
       The PMI matrix to be passed in. 
     U - (n x d dense matrix)
       The word embedding matrix.
     V - (n x d dense matrix)
       The context embedding matrix.
     lambda_1, lambda_2 - (double)
       Smoothing constants.
   Returns:
     grad_U - (n x d dense matrix)
       The gradient with respect to U. 
   Notes:
     Currently the implementation will compute the gradient in one fell 
     swoop, as the input matrix P gets larger, we will need to move to batch 
     processing to manage the data. 
-----------------------------------------------------------------------------'''
def svd_grad_U(P, U, V):
  Gram_V = grammian(V)

'''-----------------------------------------------------------------------------
    grammian(A):
      This function takes in a matrix and returns the gramian, computed in an 
      efficient fashion. 
    Inputs:
      A - (n x m dense matrix)
        The matrix to return the grammian for.
    Returns
      GramA- (m x m dense matrix)
        The grammian of A.
    TODO:
      check for under/over flow
-----------------------------------------------------------------------------'''
def grammian(A):
  (n,m) = A.shape
  GramA = np.empty((m,m))
  for i in range(n):
    ith_col = A[:,i]
    for j in range(i):
      if j == i:
        #check if iterating over range() is faster
        GramA[i,i] = reduce(lambda sum, x: x**2 + sum, ith_col, 0.0)
      else:
        entry = 0.0
        for k in range(n):
          entry += A[k,i]*A[j,k]
        GramA[i,j] = entry
        GramA[j,i] = entry
  return GramA

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