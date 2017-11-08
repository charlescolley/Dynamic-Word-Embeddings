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
  #test_function()
#  slices = []
 # tensorflow_embedding([sp.random(5,5,density=.6,format="dok"),
  #                      sp.random(5,5,density=.6,format="dok")],
  #lambda1=.01, d=2, batch_size=5, iterations=100)
 #tensorflow_SGD_test(sp.random(5,5,density=.6,format="dok"),.01,d=5,
  #                    batch_size=10,iterations=1)
  tensorflow_SGD(sp.random(10,10,density=1,format="dok"), d=5, batch_size=3)

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
    optimizer = tf.train.AdamOptimizer()
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
    PMI = tf.sparse_placeholder(tf.float32)
    #PMI = tf.SparseTensor(indices=P.keys(),values=P.values(),dense_shape=[n,n])
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
    indices = np.random.choice(range(n),1,replace=False)
    sess.run()
    print "before:", sess.run(U)
    print "after: ", sess.run(U)

    sess.run(train)

  U_res,B_res = sess.run([U,B])
  return U_res, B_res

def test_function():
  sess = tf.Session()

  np_b = np.random.rand(5)

  a = tf.get_variable("a",initializer=tf.random_uniform([4,4]))
  b = tf.get_variable("b", initializer=tf.random_uniform([2, 2]))

  init = tf.global_variables_initializer()
  sess.run(init)

  print "a", sess.run(a)
  print "b", sess.run(b)

  print "slice multiplied", sess.run(tf.matmul(a[:,:2],b))

def frobenius_diff(A, B, C):
  return tf.reduce_sum((tf.sparse_add(A,tf.matmul(B, C,transpose_b=True)))** 2)

def tf_zip(T1_list, T2_list):
  tf.TensorArray(
    tf.map_fn(lambda (x,y): tf.stack([x,y]),zip(T1_list,T2_list)))

def tensorflow_SGD(P, d, batch_size = 1):
  n = P.shape[0]
  P = P.astype(np.float32)
  sess = tf.Session()

  #initialize arrays
  total_partitions = int(ceil(n/float(batch_size)))
  PMI_section = tf.sparse_placeholder(dtype=tf.float32)
  U_segments = total_partitions * [None]


  B = tf.get_variable("B",initializer=tf.ones([d,d]))

  #define a function for instantiating a sparse subtensor from P
  def tf_P_submatrix(i,j):
    if i != total_partitions and j != total_partitions:
      P_submatrix = P[i * batch_size:(i + 1) * batch_size,
                      j * batch_size:(j + 1) * batch_size]
      shape = np.array([batch_size, batch_size])
    elif j != total_partitions:
      P_submatrix = P[ -(n % batch_size):,
                      j * batch_size:(j + 1) * batch_size]
      shape = np.array([n % batch_size, batch_size])
    elif i != total_partitions:
      P_submatrix = P[i * batch_size:(i + 1) * batch_size,
                    -(n % batch_size):]
      shape = np.array([batch_size, n % batch_size])
    else:
      P_submatrix = P[-(n % batch_size):, -(n % batch_size):]
      shape = np.array([n % batch_size, n % batch_size])
    print shape
    return (np.array(P_submatrix.keys()),
            np.array(P_submatrix.values()),
            shape)


  #create variables for rows of U
  for i in range(total_partitions-1):
    U_segments[i] = \
      tf.get_variable("U_{}".format(i),
                      initializer=tf.random_uniform([batch_size,d]))

  #set the last potentially irregular elements
  U_segments[-1] = \
    tf.get_variable(("U_{}".format(n)),
                     initializer = tf.random_uniform([n % batch_size,d]))

  #define loss functions
  loss_funcs = [None]*total_partitions**2
  with tf.name_scope("loss_functions"):
    for i in range(total_partitions):
      for j in range(total_partitions):
        loss_funcs[i*total_partitions + j] = \
          frobenius_diff(PMI_section,
                         tf.matmul(U_segments[i], B),
                         tf.matmul(U_segments[j], B))

    loss = tf.reduce_sum(loss_funcs)

  with tf.name_scope("initialization"):
    init = tf.global_variables_initializer()
    sess.run(init)

  optimizer = tf.train.GradientDescentOptimizer(.1)

  print "U_segments[0] before",sess.run(U_segments[0])

  for iter in range(1):
    for i in range(total_partitions):
      for j in range(total_partitions):
        train = optimizer.minimize(
          loss, var_list=[U_segments[i],U_segments[j]])
        print i,j#,tf_P_submatrix(i,j)
        sess.run(train,feed_dict = {PMI_section:tf_P_submatrix(i,j)})
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

