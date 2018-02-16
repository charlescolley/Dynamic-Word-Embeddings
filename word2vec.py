import tensorflow as tf
import numpy as np
from warnings import warn
from time import clock
from numpy.linalg import lstsq
from itertools import izip
from math import log, floor, ceil
from scipy.sparse.linalg import svds, LinearOperator
import scipy.sparse as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from functools import reduce
import gradients as grad
import multiprocessing as mp
from process_scipts import compute_fft
import os
import psutil
#import plotly.offline as py
#import plotly.graph_objs as go
import process_data as pd
from process_scipts import slice_multiply

def main():
  print "hello world"


'''-----------------------------------------------------------------------------
    make_semi_definite(A)
      This function takes in a matrix A and returns the matrix with any 
      eigenvectors associated with negative eigenvalues projected out, 
      this makes the matrix positive-semi definite as it will become rank 
      deficient. 
    Input:
      A - (n x n numpy or scipy matrix)
        The matrix in question to become semi-postive definite
    Returns:
      Dense matrix coming from the outer product of the eigenvectors 
      associated with the positive eigenvalues. 
    Note:
      because scipy.linalg doesn't allow for computation of the full 
      spectrum, a sparse matrix is converted to a dense matrix, this means 
      that if a matrix is too large, this function will fail. 
-----------------------------------------------------------------------------'''
def make_semi_definite(A):
  if sp.issparse(A):
    vals, vecs = np.linalg.eig(A.todense())
  else:
    vals, vecs = np.linalg.eig(A)

  for (index, eigenvalue) in enumerate(vals):
    if eigenvalue < 0:
      vals[index] = 0

  return np.dot(vecs,np.dot(np.diag(vals),vecs.T))


def make_test_tensor():
  n = 2
  m = 3
  k = 2

  a = [None] * k
  for i in range(k):
    a[i] = sp.random(n,m,density=0.0,format= 'dok')

  val = 1
  for i in range(k):
    for j in range(n):
      for l in range(m):
        a[i][j,l] = val
        val += 1

  return a

'''-----------------------------------------------------------------------------
   svd_embedding(pmi, k):
     compute a 3-dimensional embedding for a given pmi matrix and plot it 
     using a 3-d scatter plot. 
-----------------------------------------------------------------------------'''
def svd_embedding(matrix, k):
  if k == 0:
    results = svds(matrix, k=3)
  else:
    results = svds(rank_1_Update(matrix, k), k=3)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection ='3d')
  # .text(x, y, z, s, zdir=None, **kwargs
  ax.scatter(xs = results[0][:,0],ys = results[0][:,1],zs = results[0][:,2])
  plt.show()

'''-----------------------------------------------------------------------------
    plot_embeddings(embedding, words)
      this function takes in an arbitrary 3 dimensional embedding and plots 
      it using plotly's offline plotting library. if a dictionary linking the 
      indices of the words in the pmi matrix to the actual words is passed 
      in, then the plot will add the words in such that scrolling over them 
      will display the words. 
    Input:
      embedding - (n x 3 dense matrix)
        the embeddings to be plotted, ideally from pca or t-sne
      words - (dictionary)
        a dictionary linking the indices to the words they represent. keys 
        are indices, values are the text.
-----------------------------------------------------------------------------'''
def plot_embeddings(embedding, words =None):

  print embedding[:,0]
  trace1 = go.scatter3d(
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
  layout = go.layout(
    margin=dict(
      l=0,
      r=0,
      b=0,
      t=0
    )
  )
  fig = go.figure(data=data,layout = layout)
  py.plot(fig,filename='embedding.html')

'''-----------------------------------------------------------------------------
    rank_1_update(matrix, k)
       This function takes in a sparse n,m scipy matrix A and and a real value k
       and returns a LinearOperator which is of the form A + ones(n,m)*log(
       k). This can be used in the scipy svds routine in order to compute 
       add in the log rank1 update without losing the speed of a matvec in the 
       sparse case. 
    Input:
    Input:
      matrix - (n x m sparse matrix)
        the pmi matrix to use to compute the word embeddings. 
      k - (int)
        the negative sample multiple factor.
    Returns:
      LinearOperator(A + ones(n,m)*log(k))
        This just istantiates an element of the 
        scipy.sparse.linalg.LinearOperator class. Note that the mat_vec, 
        rmat_vec functions have been designed to meet the contract 
        established in the scipy class documentation. 
-----------------------------------------------------------------------------'''
def rank_1_Update(matrix, k):
  logfactor = log(k)
  n = matrix.shape[0]
  m = matrix.shape[1]
  def mat_vec(v):
    if v.shape == (m,):
      output_vec = np.empty(n)
    elif v.shape == (m,1):
      output_vec = np.empty((n,1))
    else:
      raise ValueError("non-vector passed into mat_vec, object of shape {"
                       "}".format(v.shape))

    rank_1_update = v.sum() * logfactor
    for (i,Av_i) in enumerate(matrix * v):
      output_vec[i] = Av_i + rank_1_update
    return output_vec

  def rmat_vec(v):
    if v.shape == (n,):
      output_vec = np.empty(m)
    elif v.shape == (n, 1):
      output_vec = np.empty((m, 1))
    else:
      raise ValueError("non-vector passed into mat_vec, object of shape {" \
                       "}".format(v.shape))

    rank_1_update = v.sum() * logfactor
    for (i, vTA_i) in enumerate(matrix.T * v):
      output_vec[i] = vTA_i + rank_1_update
    return output_vec

  return LinearOperator((n, m), mat_vec, rmatvec=rmat_vec)

'''-----------------------------------------------------------------------------
    mean_center(matrix)
      This function takes in a sparse scipy n x m matrix and returns a scipy 
      LinearOperator which corresponds to the matrix which has been mean 
      centered 
        MC(A) = (I_n - ones(n,n)/n)A(I_m - ones(m,m)/m)/2
      where the I_n is the n dimensional matrix identity. 
    Input:
      matrix - scipy sparse matrix
        input is assumed to be a scipy sparse matrix, the only significance 
        of it not being scipy is that the matrix vector * overloading may 
        fail, in which case this function can easily be updated to generalize to 
        another matrix format
    Returns:
      LinearOperator class instantiation
    Reference Note:
      Word, graph and manifold embedding from Markov processes
      - Tatsunori B. Hashimoto, David Alvarez-Melis, Tommi S. Jaakkola
-----------------------------------------------------------------------------'''
def mean_center(matrix):
  n = matrix.shape[0]
  m = matrix.shape[1]

  def mat_vec(v):
    if v.shape == (m,):
      output_vec = np.empty(n)
    elif v.shape == (m,1):
      output_vec = np.empty((n,1))
    else:
      raise ValueError("non-vector passed into mat_vec, object of shape {"
                       "}".format(v.shape))
    #mean center v
    mean_centered_v = np.empty(m)
    v_mean = np.mean(v)
    for (i,v_i) in enumerate(v):
      mean_centered_v[i] = (v_i - v_mean)/2

    #do the matvec
    output_vec = matrix * mean_centered_v

    #mean center the resultant vector
    output_mean = np.mean(output_vec)
    for (i,op_i) in enumerate(output_vec):
      output_vec[i] = op_i - output_mean

    return output_vec

  def rmat_vec(v):
    if v.shape == (n,):
      output_vec = np.empty(m)
    elif v.shape == (n, 1):
      output_vec = np.empty((m, 1))
    else:
      raise ValueError("non-vector passed into mat_vec, object of shape {"
                       "}".format(v.shape))
    # mean center v
    mean_centered_v = np.empty(n)
    v_mean = np.mean(v)
    for (i, v_i) in enumerate(v):
      mean_centered_v[i] = (v_i - v_mean)/2

    # do the matvec
    output_vec = matrix.T * mean_centered_v

    # mean center the resultant vector
    output_mean = np.mean(output_vec)
    for (i, op_i) in enumerate(output_vec):
      output_vec[i] = op_i - output_mean

    return output_vec

  return LinearOperator((n,m),mat_vec, rmatvec = rmat_vec)

'''-----------------------------------------------------------------------------
    matrix_power(matrix,k)
        This function takes in a matrix and a positive integer k, and returns a 
      scipy linearOperator which corresponds to the kth power of the matrix. 
    Input:
      matrix - (n x n sparse matrix) 
        assumed to be a sparse matrix because a dense matrix doesn't have 
        worry about fill in, so a matrix can simply be self multiplied and it 
        will take up the same amount of space. 
      k - (positive integer)
        the power of the matrix. k is assumed to be > 1.
    Returns:
     A_k -(scipy  LinearOperator)
       linear operator corresponding the matrix A^k.
-----------------------------------------------------------------------------'''
def matrix_power(matrix,k):
  n = matrix.shape[0]
  m = matrix.shape[1]
  if n != m:
    raise ValueError("matrix not square, shape of matrix is {}".format(
      matrix.shape))

  def mat_vec(v):
    output_vec = np.array(v)
    for i in xrange(k):
      output_vec = matrix * output_vec

    return output_vec

  def rmat_vec(v):
    output_vec = np.array(v)
    for i in xrange(k):
      output_vec = matrix.T * output_vec

    return output_vec

  A_k = LinearOperator((n,m),mat_vec, rmatvec = rmat_vec)
  return A_k

'''-----------------------------------------------------------------------------
    truncated_svd(U,S,V)
        This function takes in left and right singular vectors, and a vector of 
      singular values and returns a linear operator which corresponds to the 
      truncated svd of the matrix in question. 
    Input:
      U - (n x d numpy array)
        A matrix which corresponds to the left singular vectors.
      S - (d numpy array)
        An array corresponding to the singular values of the matrix.
      V - (optional m x d numpy array)
        An optional matrix corresponding to the right singular vectors. If no V 
        is passed in, then the matrix is assumed to be symmetric, and U will 
        be used for the singular vectors. 
    Returns:
      (Linear Operator)
        Returns the linear operator corresponding to the truncated svd
-----------------------------------------------------------------------------'''
def truncated_svd(U, S, V=None):
  if not V:
    V = U
  n = U.shape[0]
  m = V.shape[0]

  def mat_vec(x):
    output_vec = np.dot(V.T, x)
    output_vec = S * output_vec
    return np.dot(U,output_vec)

  def rmat_vec(x):
    output_vec = np.dot(U.T, x)
    output_vec = S * output_vec
    return np.dot(V,output_vec)

  return LinearOperator((n,m),mat_vec, rmatvec = rmat_vec)



'''-----------------------------------------------------------------------------
    matrix_stats(matrix)
      this function takes in a sparse matrix and returns a collection of 
      statistics about the matrix in question. data reported about the matrix 
      includes
        rows, columns, non-zeros,and singular_values
    input:
      matrix - (n x m sparse matrix)
        the matrix in question to report matrix stats about
    returns:
      stats - (dictionary)
        a dictionary of the stats to be reported back in the where the keys 
        are the listed matrix stats reported above. 
-----------------------------------------------------------------------------'''
def matrix_stats(matrix):
  stats = {}
  stats["rows"] = matrix.shape[0]
  stats["cols"] = matrix.shape[1]
  stats["singular_values"] = svds(mat_vec(matrix, 3), k=10,
                                  return_singular_vectors=False)
  return stats

'''-----------------------------------------------------------------------------
    grammian(a):
      this function takes in a matrix and returns the gramian, computed in an 
      efficient fashion. 
    inputs:
      a - (n x m dense matrix)
        the matrix to return the grammian for.
    returns
      grama- (m x m dense matrix)
        the grammian of a (a^ta).
    todo:
      check for under/over flow
-----------------------------------------------------------------------------'''
def grammian(a):
  (n,m) = a.shape
  gram_a = np.empty([m,m])
  for i in range(m):
    ith_col = a[:,i]
    for j in range(i+1):
      if j == i:
        #check if iterating over range() is faster
        gram_a[i,i] = reduce(lambda sum, x: x**2 + sum, ith_col, 0.0)
      else:
        entry = 0.0
        for k in range(n):
          entry += a[k,i]*a[k,j]
        gram_a[i,j] = entry
        gram_a[j,i] = entry
  return gram_a

def scipy_optimizer_test_func():
  a = np.random.rand(5,5)
  x = np.random.rand(5,5)

  a = a.flatten()
  x = x.flatten()

  print "a:", a
  print "x:", x

  f = lambda x: sum((x - a)**2)
  f_prime = lambda x: grad.frob_diff_grad(x,a)

  results = opt.minimize(f,x,method='bfgs',jac=f_prime)
  print results.x

'''-----------------------------------------------------------------------------
    block_partitioned_model()
      this function creates a pmi matrix which adheres to a block partition 
      graph model for a variable amount of communities of words. the function 
      will return a block diagonal matrix with constant values in the off 
      diagonal elements of the matrix within the same community.
    inputs: 
      group_word_counts - (list of ints)
        a list of the word counts of each group must be positive values.
    returns:
      pmi_matrix -(sparse dok matrix)
        the pmi matrix to test with
      word_ids - (dicitonary)
        the dictionary linking the indices to word names, this is used for 
        testing with the word_embedding_arithmetic and normalize_wordids 
        functions.
-----------------------------------------------------------------------------'''
def block_partitioned_model(group_word_counts):
  constant = 100
  n = sum(group_word_counts)
  pmi_matrix = sp.dok_matrix((n,n))

  #generate the matrix
  starting_index = 0
  for group_word_count in group_word_counts:
    for i in range(starting_index,starting_index + group_word_count):
      for j in range(starting_index,i):
        if (i != j):
          pmi_matrix[i,j] = constant
          pmi_matrix[j,i] = constant
    starting_index += group_word_count

  #generate the word ids
  word_ids = {}
  group_id = 0
  index = 0
  for group_word_count in group_word_counts:
    for i in range(group_word_count):
      word = "word_{}_group_{}".format(i,group_id)
      word_ids[word] = index
      index += 1
    group_id += 1

  return pmi_matrix, word_ids

'''-----------------------------------------------------------------------------
    tensorflow_embedding(p_list, lambda1, lambda2, d)
      this function uses the tensorflow library in order to compute an embedding
      for the words present in the pmi matrix passed in. 
    inputs:
      p_list -(n x n sparse matrix) list
        a list of the pmi matrices the embedding will be learned from.
      lambda1 - (float)
        the regularization constant multiplied to the frobenius norm of the u 
        matrix embedding.
      d - (int)
        the dimensional embedding to be learned.
      batch_size - (int)
        a positive integer which must be great than 0, and less than n.
      iterations - (int)
        the number of iterations to train on.
      results_file - (optional str)
        the file location to write the summary files to. used for running 
        tensorboard
      display_progress - (optional bool)
        updates the user in increments of 10% of how much of the training has 
        completed.
    returns:
      u_res - (n x d dense matrix)
        the d dimensional word emebedding 
      b_res - (n x d dense matrix)
        the d dimensional core tensor of the 2-tucker factorization
-----------------------------------------------------------------------------'''
def tensorflow_embedding(p_list, lambda1,lambda2, d, iterations,
                         results_file=None,
                         display_progress = False):
  if results_file:
    writer = tf.summary.filewriter(results_file)

  n = p_list[0].shape[0]
  slices = len(p_list)
  sess = tf.session()

  with tf.name_scope("loss_func"):
    lambda_1 = tf.constant(lambda1,name="lambda_1")
    lambda_2 = tf.constant(lambda2, name="lambda_2")

    u = tf.get_variable("u",initializer=tf.random_uniform([n,d], -0.1, 0.1))
    b = tf.get_variable("b",initializer=tf.ones([slices,d,d]))
    #pmi = tf.sparse_placeholder(tf.float32)

 #   indices = [(slice,i,j) for (i,j) in x.keys() for slice,x in enumerate(
  #    p_list)]

    indices = reduce(lambda x,y: x + y,[[(i,y,z) for (y,z) in p.keys()] for i,\
        p in enumerate(p_list)])
    values = reduce (lambda x,y: x + y, map(lambda x: x.values(),p_list))
    pmi = tf.sparsetensor(indices=indices, values=values,
                          dense_shape=[slices, n, n])

    ub = tf.map_fn(lambda b_k: tf.matmul(u,b_k),b)
    svd_term = tf.norm(tf.sparse_add(pmi,
      tf.map_fn(lambda ub_k: tf.matmul(-1 * ub_k, ub_k, transpose_b=True),ub)))
    fro_1 = tf.multiply(lambda_1, tf.norm(u))
    fro_2 = tf.multiply(lambda_2,tf.norm(b))
  #  fro_2 = tf.multiply(lambda_2, tf.norm(v))
  #  b_sym = tf.norm(tf.subtract(b,tf.transpose(b)))
    loss = svd_term + fro_1
    if results_file:
      tf.summary.scalar('loss',loss)
      tf.summary.tensor_summary("u",u)
      tf.summary.tensor_summary("b",b)

  with tf.name_scope("train"):
    optimizer = tf.train.adamoptimizer()
    train = optimizer.minimize(loss)

  if results_file:
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

  init = tf.global_variables_initializer()
  sess.run(init)

  print sess.run(b)
  for i in range(iterations):
    if display_progress:
      if (i % (.1*iterations)) == 0:
        print "{}% training progress".format((float(i)/iterations) * 100)

    if results_file:
      if (i % 5 == 0):
        writer.add_summary(sess.run(merged_summary),i)
    sess.run(train)

  u_res,b_res = sess.run([u,b])
  print b_res
  return u_res, b_res

def tf_submatrix(p,i_indices, j_indices):
 return tf.map_fn(lambda x: tf.gather(x, j_indices), tf.gather(p, i_indices))

'''-----------------------------------------------------------------------------
    tf_random_batch_process(p_slices, lambda1, lambda2, d, batch_size,
                            iterations, method, results_file)
      this function uses the tensorflow to compute a shared emebedding along 
      with a core tensor b in order to embedd the data in the list of pmi 
      matrices into a d dimensional real space.
    Inputs:
      p_slices -(n x n sparse dok matrix) list
        a list of the pmi matrices the embedding will be learned from.
      lambda1 - (float)
        the regularization constant multiplied to the frobenius norm of the u 
        matrix embedding.
      lambda2 - (float)
        the regularization constant multiplied to the frobenius norm of the b 
        matrix embedding.
      d - (int)
        the dimensional embedding to be learned.
      batch_size - (int)
        a positive integer which must be great than 0, and less than n, 
        used to chunk up the work of compute the gradients at each step of 
        the line search method.
      iterations - (int)
        the number of iterations to train on.
      method - (string)
        the choice of optimizer to minimize the objective function with. each 
        will be run using the randomized batch chosen at each step. options for
        the input string include
          'gd'
            gradient descent algorithm
          'ada'
            adagrad algorithm
          'adad'
            adagrad delta algorithm
          'adam'
            adam algorithm
         note that currently the parameters for each method will be set
      results_file - (optional str)
        the file location to write the summary files to. used for running 
        tensorboard
    Returns:
      u_res - (n x d dense matrix)
        the d dimensional word emebedding 
      b_res - (t x d x d dense tensor)
        the d dimensional core tensor of the 2-tucker factorization
-----------------------------------------------------------------------------'''
def tf_random_batch_process(p_slices, lambda1, lambda2, d, batch_size,
                            iterations,method, results_file = None,
                            return_loss = False, include_core = True):

  T = len(p_slices)
  n = p_slices[0].shape[0]
  record_frequency = 5
  update_messages = 1

  #ignore gpus
  os.environ["cuda_visible_devices"] = '-1'

  if results_file:
    writer = tf.summary.filewriter(results_file)

  with tf.Session(config=tf.ConfigProto(
                  log_device_placement=False)) \
       as sess:
    with tf.name_scope("loss_func"):
      u = tf.get_variable("u",dtype=tf.float32,
                          initializer=tf.ones([n,d]))

      p = tf.sparse_placeholder(dtype=tf.float32,
                                shape=np.array([batch_size, batch_size], dtype=np.int64))
      i = tf.placeholder(dtype=tf.int32,shape=np.array([batch_size,],dtype=np.int64))
      j = tf.placeholder(dtype=tf.int32,shape=np.array([batch_size,],dtype=np.int64))
      k = tf.placeholder(dtype=tf.int32)

      reg_u = lambda1 * tf.reduce_sum(tf.square(u))

      if include_core:
        b = tf.get_variable("b", dtype=tf.float32,
                            initializer=tf.ones([T, d, d]))

        b_ku_j = tf.tensordot(tf.gather(u,j),b[k],1)
        b_ku_i = tf.tensordot(tf.gather(u,i),b[k],1)

        loss_ij = tf.reduce_sum(tf.square(
          tf.sparse_add(p, tf.matmul(-1*b_ku_i, b_ku_j,
                                            transpose_b=True))))

        loss_ij_on_nil = tf.reduce_sum(tf.square(
          tf.matmul(b_ku_i,b_ku_j, transpose_b=True)))

        reg_b = lambda2 * tf.reduce_sum(tf.square(b))

        total_loss = loss_ij + reg_u + reg_b
        total_loss_on_nil = loss_ij_on_nil + reg_u + reg_b
      else:
        loss_ij = tf.reduce_sum(tf.square(
          tf.sparse_add(p, tf.matmul(-1 * tf.gather(u,i), tf.gather(u,j),
                                     transpose_b=True))))

        loss_ij_on_nil = tf.reduce_sum(tf.square(
          tf.matmul(tf.gather(u,i), tf.gather(u,j), transpose_b=True)))

        total_loss = loss_ij + reg_u
        total_loss_on_nil = loss_ij_on_nil + reg_u

      if results_file:
        total_summ = tf.summary.scalar('loss',total_loss)
        total_on_nil_summ =  tf.summary.scalar('loss_on_nil',total_loss_on_nil)
        u_summ =  tf.summary.tensor_summary("u",u)
        b_summ = tf.summary.tensor_summary("b",b)

    with tf.name_scope("train"):
      if method == 'Ada':
        optimizer = tf.train.AdadeltaOptimizer(.01)
      elif method == 'Adad':
        optimizer = tf.train.AdadeltaOptimizer()
      elif method == 'Adam':
        optimizer = tf.train.AdamOptimizer()
      elif method == 'Momen':
        optimizer = tf.train.MomentumOptimizer(learning_rate=.01)
      elif method == 'Nest':
        optimizer = tf.train.MomentumOptimizer(learning_rate=.01,
                                               use_nesterov=True)
      else:
        optimizer = tf.train.GradientDescentOptimizer(.01)
      train = optimizer.minimize(total_loss)
      train_on_nil = optimizer.minimize(total_loss_on_nil)

    if results_file:
      writer.add_graph(sess.graph)


    init = tf.global_variables_initializer()
    sess.run(init)

    if return_loss:
      loss = []


    for step in range(1,iterations+1):

      #update user
      if not (step % (iterations/update_messages)):
        print "finished {}% steps completed".format(
          (100*float(step)/iterations))
      tf_i = np.random.choice(n,size=batch_size,replace=False)
      tf_j = np.random.choice(n,size=batch_size,replace=False)
      tf_k = 0 if T == 1 else np.random.choice(T,size=1)[0]
      sub_matrix_p = (p_slices[tf_k])[tf_i][:,tf_j]

      #switches to different lo2*T * (n * i + (col_offset + j)) + 2*kss function if sparse tensor is empty
      if sub_matrix_p.nnz:
        params = \
          {p: (sub_matrix_p.keys(), sub_matrix_p.values(), [batch_size, batch_size]),
                  i: tf_i, j: tf_j,k:tf_k}

        sess.run(train,feed_dict =params)
        if return_loss:
          if step % 1 == 0:
            loss_func_val = 0.0
            for t in range(T):
              params = {p: (p_slices[t].keys(), p_slices[t].values(),[n, n]),
                        i: range(n), j: range(n), k: t}
              loss_func_val += sess.run(loss_ij,feed_dict=params)
            loss_func_val += sess.run(reg_u)
            if include_core:
              loss_func_val += sess.run(reg_b)
            loss.append(loss_func_val)
      else:
        params = {i: tf_i, j: tf_j,k:tf_k}
        sess.run(train_on_nil, feed_dict=params)

      if results_file:
        if not step % record_frequency:
          writer.add_summary(sess.run(u_summ,feed_dict=params))
          writer.add_summary(sess.run(b_summ,feed_dict=params))
          if sub_matrix_p.nnz:
            writer.add_summary(sess.run(total_summ,feed_dict=params), step)
          else:
            writer.add_summary((sess.run(total_on_nil_summ, feed_dict=params)))

    if results_file:
      writer.close()

    u_res = sess.run(u)
    b_res = sess.run(b) if include_core else None

  loss = loss if return_loss else None
  return u_res,b_res, loss

'''-----------------------------------------------------------------------------
    evaluate_embedding(u.b,lambda1,lambda2,years,p_slices)
      this function takes in a computed u and b from a given run and returns 
      the value of the loss function at that point and the frobenius norm of 
      the jacobian. this function sets up a tensorflow computation graph to 
      compute both. this function must be run in the main folder with all the 
      pmi matrices in order to have access to all the relevant files. 
    Input:
      u - (n x d dense matrix)
        the shared embedding u
      b - (d x d dense matrix)
        the core tensor slices 
      lambda1 - (float)
        the regularizer term of u
      lambda2 - (float)
        the regularizer term for b
      years  - (int list)
        a list of years that the embedding is for.
      method - (string)
        the type of optimizer run, used for computing the gradients
      p_slices - (optional list of sparse dok matrices)
        a list of sparse matrices to use if the embeddings come from 
        somewhere other than the pmi matrices.
    Returns:
      loss_func_val -(float)
        the value of the loss function
      u_grad_fro_norm-(float)
        the frobenius norm of the gradient with respect to u
      b_grad_fro_norm - (float)
        the frobenius norm of the gradient with respect to ub
-----------------------------------------------------------------------------'''
def evaluate_embedding(u,b,lambda1,lambda2, years,p_slices =None):

  #load in the relevant time slices if none passed in
  if p_slices == None:
    pmi_matrices = []
    word_ids = []
    for year in years:
      file = "wordpairpmi_" + str(year) + ".csv"
      pmi, ids = pd.read_in_pmi(file)
      pmi_matrices.append(pmi)
      word_ids.append(ids)

    if len(years) > 1:
      shared_id = pd.normalize_wordids(pmi_matrices,ids)
  else:
    pmi_matrices = p_slices

  slice_wise_loss_funcs = []

  with tf.Session() as sess:
    tf_u = tf.get_variable("u_tf",initializer=u)
    if b:
      tf_b = tf.get_variable("b_tf", initializer=b)

    tf_p = []
    for i in range(len(years)):
      tf_p.append(tf.SparseTensorValue(pmi_matrices[i].keys(),pmi_matrices[
        i].values(),[pmi_matrices[i].shape[0], pmi_matrices[i].shape[1]]))
    for i in range(len(years)):
      if b:
        ub = tf.matmul(tf_u, tf_b[i])
        loss_func_i = tf.reduce_sum(tf.square(
          tf.sparse_add(tf_p[i], tf.matmul(-1 * ub, ub, transpose_b=True))))
      else:
        loss_func_i = tf.reduce_sum(tf.square(
        tf.sparse_add(tf_p[i], tf.matmul(-1 * u, u, transpose_b=True))))

      slice_wise_loss_funcs.append(loss_func_i)

    reg_u = lambda1 * tf.reduce_sum(tf.square(u))
    if b:
      reg_b = lambda2 * tf.reduce_sum(tf.square(b))
      total_loss_func = tf.reduce_sum(slice_wise_loss_funcs) + reg_u + reg_b
    else:
      total_loss_func = tf.reduce_sum(slice_wise_loss_funcs) + reg_u

    optimizer = tf.train.GradientDescentOptimizer(.01)

    init = tf.global_variables_initializer()
    sess.run(init)

    loss_val = sess.run(total_loss_func)
    u_grad_fro_norm = sess.run(tf.reduce_sum(tf.square(
      optimizer.compute_gradients(total_loss_func,tf_u)[0])))
    if b:
      b_grad_fro_norm = sess.run(tf.reduce_sum(tf.square(
        optimizer.compute_gradients(total_loss_func,tf_b)[0])))
    else:
      b_grad_fro_norm = None

    return loss_val, u_grad_fro_norm, b_grad_fro_norm

'''-----------------------------------------------------------------------------
    project_onto_positive_eigenspaces(a)
      this function takes in a np 2d array and returns the dense matrix with the
      eigenspaces associated with eigenvalues < 0 removed. 
-----------------------------------------------------------------------------'''
def project_onto_positive_eigenspaces(a):
  vals, vecs = np.linalg.eigh(a)
  positive_eigs = filter(lambda x: vals[x] > 0, range(a.shape[0]))
  submatrix = vecs[np.ix_(range(a.shape[0]), positive_eigs)]
  return np.dot(submatrix,(vals[positive_eigs]*submatrix).t)


'''-----------------------------------------------------------------------------
    mode_3_fft(a, max_cores)
      this function takes in a 3rd order real tensor representation and 
      computes the fft along the mode 3 fibers. The function assumes a real 
      input and thus returns an array which has the 0th and positive 
      frequency coefficients rather than the entire spectrum. The columns of 
      the tensor are divided amongst the physical cores of on the system 
      being run in order to minimize run time. 
    Input:
      A - (list of square dok_matrices float64)
        the tensor representation is assumed to be a list of n x n sparse dok 
        matrices of 64 bit floating point numbers. The length of the list 
        denotes the mode 3 dimension of the tensor and is represented by T. 
      max_cores - optional (int)
        The number of cores to run the computation with. Note that this is 
        taken over the minimum of the dimension of the matrices, and the 
        number of physical cores on the system. If none is passed in, 
        the default is set to a excessively large number (in terms of cores) 
        in order to default to min(available cores, n), in practice this 
        should be all available cores on the machine. 
    Returns:
      fft_A - (n x n n T' ndarray of np.complex)
        fft_A is a dense tensor in the form of an ndarray which has the 0 
        frequency terms and the positive frequency fft coefficients along the 
        3rd mode.
        
-----------------------------------------------------------------------------'''
def mode_3_fft(A, max_cores=None):
  if not max_cores:
    max_cores = 100000000 #unreasonably large number of cores

  n = A[0].shape[0]
  T = len(A)

  a = rotate_tensor(A)

  #shared array must be float64, will be cast to complex128 in processes
  fft_p = mp.RawArray('d',int(2*n*n*(1 + floor(T/2.0))))

  #set up cores to compute the fft along 3rd mode
  jobs = []
  core_count = psutil.cpu_count(False) #don't count virtual cores
  process_count = min(core_count,max_cores,n)

  slices_per_process = int(ceil(n / float(process_count)))

  for i in xrange(process_count):
    start = i*slices_per_process
    end = min((i+1)*slices_per_process, n)
    print "core {} gets".format(i),a[start:end]
    p = mp.Process(target=compute_fft, name=i + 1,
                   args=(a[start:end],fft_p,start))
    jobs.append(p)
    p.start()

  #wait for processes to finish running
  for p in jobs:
    p.join()

  for val in fft_p:
    print val

  fft_A = np.ndarray(buffer = fft_p,shape=(n,n,1 + int(floor(T/2.0))),
                     order='C',dtype=np.complex)
  return fft_A

'''-----------------------------------------------------------------------------
    flattened_svd(a,k,used_mean_centered,parallel)
      this function takes in a tensor in the form of a list of sparse dok 
      matrices, flattens them along their first mode and then computes the top 
      k left singular vectors and uses this for an embedding and an 
      approximation of the core tensor by multiplying each slice from the 
      right with u^t. 
    Input:
      a - (list of sparse dok matrices) 
        the tensor representation of the day
      k - (int)
        the number of singular vectors to compute
      LO_type - (optional string)
        a string indicting which type of linear operator should be applied to 
        each of the slices before creating a linear operator which represents 
        the mode one flattening. For documentation on each of the options, 
        see the documentation of create_flattened_Linear_Operators(...). 
        Options:
          mean_center, power, already_applied
      use_V - (optional bool)
        a boolean indicating whether or not to use the right singular vectors 
        corresponding to the mode-1 flattening. 
      years_used - (optional list of ints)
        The list of years used, which are used to load in the singular 
        vectors of each of the individual time slices to form the core 
        tensor. This must not be empty if use_V is true, if it is, a warning 
        will be displayed and use_V will be set to false.
    Return:
      U - (n x k ndarray)
        the shared embedding in question
      sigma - (n array)
        the sigular values of the singular vectors
      b - (t x k x k ndarray)
        the approximated core tensor
-----------------------------------------------------------------------------'''
def flattened_svd(A,k, LO_type = None,use_V = False,years_used = None):

  if use_V:
    if not years_used:
      warn("use_V is True, but no years are passed in, defaulting to use_V = "
           "False\n")
      use_V = False


  T = len(A)
  if LO_type:
    print "using linear operator {}".format(LO_type)
    A_1 = create_flattened_Linear_Operators(A,LO_type)
  else:
    print "using sparse arrays"
    A_1 = flatten(A)

  b = np.ndarray((T, k, k))


  if use_V:
    U, sigma, VT = svds(A_1, k)
    n = U.shape[0]
    for t in range(T):
 #     filename = "full_svd/full_wordPairPMI_" +str(years_used[t])+ "_U.npy"
  #    U_t = np.load(filename)
      b[t] = np.dot(VT[:,t*n:(t+1)*n].T,VT[:,t*n:(t+1)*n])

  else:
    U, sigma, _ = svds(A_1, k=k, return_singular_vectors="u")
    for t in range(T):
      A[t] = A[t].tocsr()
      b[t] = np.dot(U.T, A[t].dot(U))




  return U, sigma, b

'''-----------------------------------------------------------------------------
    create_flattened_Linear_Operators(slices, LO_type)
        This function takes in a list of sparse matrices and a string, and 
      returns a linear Operator which corresponds to the mode one flattened 
      matrix where each slice is first turned into a linear operator. The 
      string passed in will determine which type of linear operator is 
      applied to each slice. 
    Input:
      slices - (list of sparse matrices)
        The tensor slices to produce the the flattened linear operator with. 
        This list is assumed to be at least of length 2. 
      LO_type -(string)
        The string which will determine which type of linear operator should 
        be created from each slice.
        Options:
          mean_center 
            - mean centers any incoming vectors, and the resulting output 
              vector from the linear mapping. 
          power 
            - powers each one of the matrices according to the power k, 
              passed in 
          already_applied 
            - the list of slices are already a list of linear operators.
      k - (optional int)
        This integer is only used if the LO_type is power. This k corresponds to
        the power of k each matrix will be powered to.  
    Return:
      flattened_LO - (LinearOperator)
        the (n,mT) linear operator corresponding to the mode one flattening 
        of all the tensor slices with the respective linear operator applied 
        to them. 
    Note:
      as more general linear operators are implented and added in, it will be 
      useful to write something that takes in the parameters of the given LO 
      and apply them in the needed locations, instead of having a collection 
      of unused optional arguments. 
-----------------------------------------------------------------------------'''
def create_flattened_Linear_Operators(slices, LO_type, k=2):
  n = slices[0].shape[0]
  m = slices[0].shape[1]
  T = len(slices)

  # apply the linear operators to each slice
  LO_slices = []
  for slice in slices:
    if LO_type == "mean_center":
      LO_slices.append(mean_center(slice))
    elif LO_type == "power":
      LO_slices.append(matrix_power(slice,k))
    elif LO_type == "already_applied":
      LO_slices = slices
    else:
      raise ValueError("invalid Linear Operator type")

  def mat_vec(v):
    if v.shape == (m*T,):
      output_vec = np.empty(n)
    elif v.shape == (m*T,1):
      output_vec = np.empty((n,1))
    else:
      raise ValueError("non-vector passed into mat_vec, object of shape {"
                       "}".format(v.shape))
    output_vec = LO_slices[0] * v[0:m]
    for t in range(1,T):
      output_vec = output_vec + LO_slices[t] * v[t*m:(t+1)*m]

    return output_vec

  def rmat_vec(v):
    if v.shape == (n,):
      output_vec = np.empty(m*T)
    elif v.shape == (n, 1):
      output_vec = np.empty((m*T, 1))
    else:
      raise ValueError("non-vector passed into mat_vec, object of shape {"
                       "}".format(v.shape))
    for t in range(T):
      output_vec[t*m:(t+1)*m] = LO_slices[t].rmatvec(v)

    return output_vec
  return LinearOperator((n,m*T), mat_vec, rmatvec= rmat_vec)

'''-----------------------------------------------------------------------------
    flatten(a)
      this function takes in a list of sparse dok scipy matrices and returns 
      the mode 1 flattened tensor in a csr format. 
    Inputs:
      a - (list of sparse dok matrices)
        the tensor representation of the data
    Returns:
      a_1 - (csr sparse matrix)
        the mode 1 flattening of the tensor a
    
-----------------------------------------------------------------------------'''
def flatten(a):
  T = len(a)
  n = a[0].shape[0]
  a_1 = sp.dok_matrix((n,T*n))

  if str(type(a[0])) == "<class 'scipy.sparse.coo.coo_matrix'>":
    for t in range(T):
      for i, j, nnz in izip(a[t].row, a[t].col, a[t].data):
        a_1[i, j + n*t] =  nnz
  else:
    for t in range(T):
      for ((i,j), nnz) in a[t].iteritems():
        a_1[i,j + n*t] = nnz

  return a_1.tocsr()

'''-----------------------------------------------------------------------------
    rotate_tensor(a)
      this function takes in a list of n x n sparse matrices representing a 
      n x n x k tensor and returns a list of n x k sparse matrices which 
      represent a n x k x n tensor
    Input:
      a - a list of (n x n) sparse dok matrices  
    Note:
      assuming that the keys and values of each dok sparse matrix are 
      rnadomly ordered, but have the same ordering.
-----------------------------------------------------------------------------'''
def rotate_tensor(a):

  n = a[0].shape[0]
  m = a[0].shape[1]
  slice_count = len(a)
  rotated_a = [None] * m

  #initialize empty sparse matrices
  for j in range(m):
    rotated_a[j] = sp.dok_matrix((n,slice_count))

  #copy all non-zeros into their appropriate place in the rotated matrix
  for k in range(slice_count):
    for ((i,j),value) in a[k].items():
      rotated_a[j][i,k] = value

  return rotated_a


if __name__ == "__main__":
    main()

