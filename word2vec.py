import tensorflow as tf
import numpy as np
from math import log, ceil
from scipy.sparse.linalg import svds, LinearOperator
import scipy.sparse as sp
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.decomposition import TruncatedSVD
#import plotly.offline as py
#import plotly.graph_objs as go
import process_data as pd
from sklearn.manifold import TSNE

def main():
  slices = []
  tensorflow_embedding([sp.random(5,5,density=.6,format="dok"),
                        sp.random(5,5,density=.6,format="dok")],
  lambda1=.01, d=2, batch_size=5, iterations=100)
 # tensorflow_SGD_test(sp.random(5,5,density=.6,format="dok"),.01,d=5,
  #                    batch_size=10,iterations=1)
  #tensorflow_SGD(sp.random(10,10,density=.6,format="dok"), d=5, batch_size=3)

'''-----------------------------------------------------------------------------
   svd_embedding(pmi, k):
     compute a 3-dimensional embedding for a given pmi matrix and plot it 
     using a 3-D scatter plot. 
-----------------------------------------------------------------------------'''
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
    plot_embeddings(embedding, words)
      This function takes in an arbitrary 3 dimensional embedding and plots 
      it using plotly's offline plotting library. If a dictionary linking the 
      indices of the words in the PMI matrix to the actual words is passed 
      in, then the plot will add the words in such that scrolling over them 
      will display the words. 
    Input:
      embedding - (n x 3 dense matrix)
        the embeddings to be plotted, ideally from PCA or t-SNE
      words - (dictionary)
        a dictionary linking the indices to the words they represent. keys 
        are indices, values are the text.
-----------------------------------------------------------------------------'''
def plot_embeddings(embedding, words =None):

  print embedding[:,0]
  trace1 = go.Scatter3d(
    x = embedding[:,0],
    y = embedding[:, 1],
    z = embedding[:, 2],

    mode = 'text',
    hoverinfo = words.values(),
           marker = dict(
      color='rgb(127, 127, 127)',
      size=12,
      symbol='circle',
      line=dict(
        color='rgb(204, 204, 204)',
        width=1
      ),

      opacity=0.9
    )
  )
  data = [trace1]
  layout = go.Layout(
    margin=dict(
      l=0,
      r=0,
      b=0,
      t=0
    )
  )
  fig = go.Figure(data=data,layout = layout)
  py.plot(fig,filename='embedding.html')

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
        
     which evaluates to (P + I\lambda_2)V - U(I\lambda_1 + V^TV). The function 
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
def svd_grad_U(P, U, V, lambda_1, lambda_2):
  (n, d) = U.shape
  Gram_V = grammian(V)
  if lambda_1 != 0:
    for i in range(d):
      Gram_V[i,i] += lambda_1
  UVTV = np.dot(U, Gram_V)

  #temporarily add in the lambda_2 term to P
  if lambda_2 != 0.0:
    for i in range(n):
      P[i,i] += lambda_2
  PV = P * V
  #remove diagonal terms
  if lambda_2 != 0:
    for i in range(n):
      P[i,i] -= lambda_2
  return PV + UVTV

'''-----------------------------------------------------------------------------
    grammian(A):
      This function takes in a matrix and returns the gramian, computed in an 
      efficient fashion. 
    Inputs:
      A - (n x m dense matrix)
        The matrix to return the grammian for.
    Returns
      GramA- (m x m dense matrix)
        The grammian of A (A^TA).
    TODO:
      check for under/over flow
-----------------------------------------------------------------------------'''
def grammian(A):
  (n,m) = A.shape
  Gram_A = np.empty([m,m])
  for i in range(m):
    ith_col = A[:,i]
    for j in range(i+1):
      if j == i:
        #check if iterating over range() is faster
        Gram_A[i,i] = reduce(lambda sum, x: x**2 + sum, ith_col, 0.0)
      else:
        entry = 0.0
        for k in range(n):
          entry += A[k,i]*A[k,j]
        Gram_A[i,j] = entry
        Gram_A[j,i] = entry
  return Gram_A

'''-----------------------------------------------------------------------------
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
-----------------------------------------------------------------------------'''
def build_loss_function(word_count_matrix, word_count, k):
  print "TODO"

'''-----------------------------------------------------------------------------
    tensorflow_embedding(P_list, lambda1, lambda2, d)
      This function uses the tensorflow library in order to compute an embedding
      for the words present in the PMI matrix passed in. 
    Inputs:
      P_list -(n x n sparse matrix) list
        a list of the PMI matrices the embedding will be learned from.
      lambda1 - (float)
        the regularization constant multiplied to the frobenius norm of the U 
        matrix embedding.
      d - (int)
        the dimensional embedding to be learned.
      batch_size - (int)
        a positive integer which must be great than 0, and less than n.
      iterations - (int)
        the number of iterations to train on.
      results_file - (optional str)
        the file location to write the summary files to. Used for running 
        tensorboard
      display_progress - (optional bool)
        updates the user in increments of 10% of how much of the training has 
        completed.
    Returns:
      U_res - (n x d dense matrix)
        the d dimensional word emebedding 
      B_res - (n x d dense matrix)
        the d dimensional core tensor of the 2-tucker factorization
-----------------------------------------------------------------------------'''
def tensorflow_embedding(P_list, lambda1,lambda2, d, iterations,
                         results_file=None,
                         display_progress = False):
  if results_file:
    writer = tf.summary.FileWriter(results_file)

  n = P_list[0].shape[0]
  slices = len(P_list)
  sess = tf.Session()

  with tf.name_scope("loss_func"):
    lambda_1 = tf.constant(lambda1,name="lambda_1")
    lambda_2 = tf.constant(lambda2, name="lambda_2")

    U = tf.get_variable("U",initializer=tf.random_uniform([n,d], -0.1, 0.1))
    B = tf.get_variable("B",initializer=tf.ones([slices,d,d]))
    #PMI = tf.sparse_placeholder(tf.float32)

 #   indices = [(slice,i,j) for (i,j) in x.keys() for slice,x in enumerate(
  #    P_list)]

    indices = reduce(lambda x,y: x + y,[[(i,y,z) for (y,z) in P.keys()] for i,\
        P in enumerate(P_list)])
    values = reduce (lambda x,y: x + y, map(lambda x: x.values(),P_list))
    PMI = tf.SparseTensor(indices=indices, values=values,
                          dense_shape=[slices, n, n])


    UB = tf.map_fn(lambda B_k: tf.matmul(U,B_k),B)
    svd_term = tf.norm(tf.sparse_add(PMI,
      tf.map_fn(lambda UB_k: tf.matmul(-1 * UB_k, UB_k, transpose_b=True),UB)))
    fro_1 = tf.multiply(lambda_1, tf.norm(U))
    fro_2 = tf.multiply(lambda_2,tf.norm(B))
  #  fro_2 = tf.multiply(lambda_2, tf.norm(V))
  #  B_sym = tf.norm(tf.subtract(B,tf.transpose(B)))
    loss = svd_term + fro_1
    if results_file:
      tf.summary.scalar('loss',loss)
      tf.summary.tensor_summary("U",U)
      tf.summary.tensor_summary("B",B)

  with tf.name_scope("train"):
    optimizer = tf.train.AdagradOptimizer(.01)
    train = optimizer.minimize(loss)

  if results_file:
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

  init = tf.global_variables_initializer()
  sess.run(init)

  print sess.run(B)
  for i in range(iterations):
    if display_progress:
      if (i % (.1*iterations)) == 0:
        print "{}% training progress".format((float(i)/iterations) * 100)

    if results_file:
      if (i % 5 == 0):
        writer.add_summary(sess.run(merged_summary),i)
    sess.run(train)

  U_res,B_res = sess.run([U,B])
  print B_res
  return U_res, B_res


def tensorflow_SGD_test(P, lambda1, d, batch_size, iterations,
                         results_file=None,
                         display_progress = False):
  if results_file:
    writer = tf.summary.FileWriter(results_file)

  n = P.shape[0]
  sess = tf.Session()

  with tf.name_scope("loss_func"):
    lambda_1 = tf.constant(lambda1,name="lambda_1")
    U = tf.get_variable("U",initializer=tf.random_uniform([n,d], -0.1, 0.1))
    B = tf.get_variable("B",initializer=tf.ones([d,d]))
    #PMI = tf.sparse_placeholder(tf.float32)
    PMI = tf.SparseTensor(indices=P.keys(),values=P.values(),dense_shape=[n,n])
    UB = tf.matmul(U,B)
    svd_term = tf.norm(tf.sparse_add(PMI,tf.matmul(-1 * UB, UB,\
                                                          transpose_b=True)))
    fro_1 = tf.multiply(lambda_1, tf.norm(U))

    loss = svd_term + fro_1
    if results_file:
      tf.summary.scalar('loss',loss)
      tf.summary.tensor_summary("U",U)
      tf.summary.tensor_summary("B",B)

  with tf.name_scope("train"):
    #optimizer = tf.train.AdagradOptimizer(.01)
    optimizer = tf.train.GradientDescentOptimizer(.01)
    train = optimizer.minimize(loss)

  if results_file:
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

  init = tf.global_variables_initializer()
  sess.run(init)
  indices = [0,1]
  for i in range(iterations):
    if display_progress:
      if (i % (.1*iterations)) == 0:
        print "{}% training progress".format((float(i)/iterations) * 100)

    if results_file:
      if (i % 5 == 0):
        writer.add_summary(sess.run(merged_summary),i)

    mask = [False]*n
    mask[2] = True
    U_sottf.stop_gradient(tf.boolean_mask(U,mask))
    print sess.run(optimizer.compute_gradients(loss,U))
    print "before:", sess.run(U)
    print "after: ", sess.run(U)

    sess.run(train)

  U_res,B_res = sess.run([U,B])
  return U_res, B_res


def frobenius_diff(A, B):
  return tf.reduce_sum((A-tf.matmul(B, B,transpose_b=True))** 2)

def tensorflow_SGD(P, d, batch_size = 1):
  n = P.shape[0]
  sess = tf.Session()

  partition_size = tf.constant([batch_size, d],dtype=tf.float32)


  #initialize arrays
  total_partitions = int(ceil(n/float(batch_size)))
  PMI_segments = total_partitions * [None]
  U_segments = total_partitions * [None]


  #create variables for rows of U and sections of P
  for i in range(total_partitions-1):
    print i, len(U_segments)
    U_segments[i] = \
      tf.get_variable("U_{}".format(i),
                      initializer=tf.random_uniform([batch_size,d]))
    tf.random_uniform([])
    P_submatrix = P[i*batch_size:(i+1)*batch_size,:]
    PMI_segments[i] = \
      tf.SparseTensor(
        indices =np.array(P_submatrix.keys()),
        values=np.array(P_submatrix.values()),
        dense_shape=np.array([batch_size,d]))

  #set the last potentially irregular elements
  U_segments[-1] = \
    tf.get_variable(("U_{}".format(n)),
                     initializer = tf.random_uniform([n % batch_size,d]))


  P_submatrix = P[-(n % batch_size):-1,:]
  PMI_segments[-1] = \
    tf.SparseTensor(
      indices=P_submatrix.keys(), values=P_submatrix.values(),
      dense_shape=partition_size)

  B = tf.get_variable("B",initializer=tf.ones([d,d]))

  #define loss functions
  with tf.name_scope("loss_functions"):
    segment_loss_func = lambda Pair: frobenius_diff(Pair[0], Pair[1] * B)
    loss = tf.reduce_sum(
      tf.foldr(segment_loss_func, tf.TensorArray(tf.stack(PMI_segments,U_segments)),\
               shape = [total_partitions,batch_size,d],\
               dtype=tf.float32))



  optimizer = tf.train.GradientDescentOptimizer(.1)
  train = optimizer.minimize(loss)
  sess = tf.Session()

  with tf.name_scope("initialization"):
    init = tf.global_variables_initializer()
    sess.run(init)


  print "PMI[0] before",sess.run(U_segments[0])


  for i in range(150):
    grad = optimizer.compute_gradients(loss,var_list=U_segments[0])
    print "grad_U_{}:".format(i),sess.run(grad)
    sess.run(U_segments[i].assign( U_segments[i] - .1*grad[0]))
    print "x after",sess.run(U_segments[i])


'''-----------------------------------------------------------------------------
    project_onto_positive_eigenspaces(A)
      This function takes in a np 2d array and returns the dense matrix with the
      eigenspaces associated with eigenvalues < 0 removed. 
-----------------------------------------------------------------------------'''
def project_onto_positive_eigenspaces(A):
  vals, vecs = np.linalg.eigh(A)
  positive_eigs = filter(lambda x: vals[x] > 0, range(A.shape[0]))
  submatrix = vecs[np.ix_(range(A.shape[0]), positive_eigs)]
  return np.dot(submatrix,(vals[positive_eigs]*submatrix).T)


'''-----------------------------------------------------------------------------
   a tensorflow helper function used to only compute certain gradients. 
   
   source - https://github.com/tensorflow/tensorflow/issues/9162
-----------------------------------------------------------------------------'''
def entry_stop_gradients(target, mask):
  mask_h = tf.logical_not(mask)

  mask = tf.cast(mask, dtype=target.dtype)
  mask_h = tf.cast(mask_h, dtype=target.dtype)

  return tf.stop_gradient(mask_h * target) + mask * target


#def t_svd()

if __name__ == "__main__":
    main()

