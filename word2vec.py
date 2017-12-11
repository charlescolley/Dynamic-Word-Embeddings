import tensorflow as tf
import numpy as np
from time import clock
from numpy.linalg import lstsq
from math import log, ceil
from scipy.sparse.linalg import svds
import scipy.sparse as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from functools import reduce
import gradients as grad
import multiprocessing as mp
from ctypes import c_double
from process_scipts import compute_fft
import os
#import plotly.offline as py
#import plotly.graph_objs as go
import process_data as pd
from process_scipts import slice_multiply

def main():
  T = 5
  n = 100
  slices = []
  for t in range(T):
    matrix = sp.dok_matrix((n,n))
    for i in range(n):
      matrix[i,i] = i
    slices.append(matrix)

  U,B = flattened_svd(slices,10)

  print U, B

'''
  lambda1 = .01
  lambda2 = .01
  d = 10
  batch_size = 40
  iterations = 1000
  method = 'adam'

  p, id = block_partitioned_model([10,15])
  u,b = tf_random_batch_process([p], lambda1, lambda2, d, 25,
                                iterations,method,include_core=False)
  loss_val, u_grad_fro_norm, b_grad_fro_norm =  \
    evaluate_embedding(u,b,lambda1,lambda2,[213212],[p])
  print loss_val, u_grad_fro_norm, b_grad_fro_norm
'''

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
    results = svds(mat_vec(matrix, k), k=3)

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
    input:
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
    mat_vec(matrix, vector)
       this function produces an anonymous function to be used as a linear 
       operator in the scipy svd routine.
    input:
      matrix - (n x m sparse matrix)
        the pmi matrix to use to compute the word embeddings. 
      k - (int)
        the negative sample multiple factor.
    returns:
      mat_vec - (m-vec -> n-vec)
        an anonymous function which works as an o(m) linear operator which 
        adds a rank 1 update to the pmi matrix.   (m - log(k))
    notes:
      unclear if the numpy sum function has numerical instability issues. 
-----------------------------------------------------------------------------'''
def mat_vec(matrix, k):
  logfactor = log(k)
  n = matrix.shape[0]
  m = matrix.shape[1]
  mat_vec = lambda v: (matrix * v) + (np.ones(n) * v.sum() * logfactor)
  rmat_vec = lambda v: (matrix.t * v) + (np.ones(m) * v.sum() * logfactor)
  return linearoperator((n, m), mat_vec, rmatvec=rmat_vec)

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
    build_objective_functions(word_count_matrix, k)
      this function takes in a n x m matrix with the scaled number of times a 
      word appears within the context c (#(w,c)) and returns a lambda 
      function which computes the loss function and a gradient function. the 
      function will 
    inputs:
      word_count_matrix - (sparse n x m matrix)
        a scipy sparse matrix which the (i,j)th entry corresponds to #(w_i,
        c_j)|d|. here |d| denotes the number of words in the text corpus. 
      word_count - (dictionary)
        a dictionary which has the word counts for a text corpus. 
      k - int
        the negative sampling rate, creates k fake samples which help prevent 
        unform distriutions from arising. samples are created from a unigram 
        distriution. p((w,c)) = #(w)*#(c)^{3/4}/z where z is a normalizing 
        constant. 
    returns:
      loss_func - (lambda func)
        an anonymous function which has the negated word2vec objective 
        function (which is typically maximized). 
        \sum_{(w,c) \in d} (log(softmax(v_c,v_w)) - k 
      gradient - (lambda func)
        an anonymous function which has the gradient of the 
-----------------------------------------------------------------------------'''
def build_loss_function(word_count_matrix, word_count, k):
  print "todo"

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
    inputs:
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
    returns:
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

  with tf.session(config=tf.configproto(
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
        total_on_nil_summ =   tf.summary.scalar('loss_on_nil',total_loss_on_nil)
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

      #switches to different loss function if sparse tensor is empty
      if sub_matrix_p.nnz:
        params = \
          {p: (sub_matrix_p.keys(), sub_matrix_p.values(), [batch_size, batch_size]),
                  i: tf_i, j: tf_j,k:tf_k}

        sess.run(train,feed_dict =params)
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

  return u_res,b_res

'''-----------------------------------------------------------------------------
    evaluate_embedding(u.b,lambda1,lambda2,years,p_slices)
      this function takes in a computed u and b from a given run and returns 
      the value of the loss function at that point and the frobenius norm of 
      the jacobian. this function sets up a tensorflow computation graph to 
      compute both. this function must be run in the main folder with all the 
      pmi matrices in order to have access to all the relevant files. 
    input:
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
    returns:
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

def frobenius_diff(a, b, c):
  return tf.reduce_sum((tf.sparse_add(a,tf.matmul(b, c,transpose_b=True)))** 2)

def tf_zip(t1_list, t2_list):
  tf.tensorArray(
    tf.map_fn(lambda (x,y): tf.stack([x,y]),zip(t1_list,t2_list)))

def tensorflow_sgd(p, d, batch_size = 1):
  n = p.shape[0]
  p = p.astype(np.float32)
  sess = tf.Session()

  #initialize arrays
  total_partitions = int(ceil(n/float(batch_size)))
  pmi_section = tf.sparse_placeholder(dtype=tf.float32)
  u_segments = total_partitions * [None]


  b = tf.get_variable("b",initializer=tf.ones([d,d]))

  #define a function for instantiating a sparse subtensor from p
  def tf_p_submatrix(i,j):
    if i != total_partitions and j != total_partitions:
      p_submatrix = p[i * batch_size:(i + 1) * batch_size,
                      j * batch_size:(j + 1) * batch_size]
      shape = np.array([batch_size, batch_size])
    elif j != total_partitions:
      p_submatrix = p[ -(n % batch_size):,
                      j * batch_size:(j + 1) * batch_size]
      shape = np.array([n % batch_size, batch_size])
    elif i != total_partitions:
      p_submatrix = p[i * batch_size:(i + 1) * batch_size,
                    -(n % batch_size):]
      shape = np.array([batch_size, n % batch_size])
    else:
      p_submatrix = p[-(n % batch_size):, -(n % batch_size):]
      shape = np.array([n % batch_size, n % batch_size])
    print shape
    return (np.array(p_submatrix.keys()),
            np.array(p_submatrix.values()),
            shape)


  #create variables for rows of u
  for i in range(total_partitions-1):
    u_segments[i] = \
      tf.get_variable("u_{}".format(i),
                      initializer=tf.random_uniform([batch_size,d]))

  #set the last potentially irregular elements
  u_segments[-1] = \
    tf.get_variable(("u_{}".format(n)),
                     initializer = tf.random_uniform([n % batch_size,d]))

  #define loss functions
  loss_funcs = [None]*total_partitions**2
  with tf.name_scope("loss_functions"):
    for i in range(total_partitions):
      for j in range(total_partitions):
        loss_funcs[i*total_partitions + j] = \
          frobenius_diff(pmi_section,
                         tf.matmul(u_segments[i], b),
                         tf.matmul(u_segments[j], b))

    loss = tf.reduce_sum(loss_funcs)

  with tf.name_scope("initialization"):
    init = tf.global_variables_initializer()
    sess.run(init)

  optimizer = tf.train.GradientDescentOptimizer(.1)

  print "u_segments[0] before",sess.run(u_segments[0])

  for iter in range(1):
    for i in range(total_partitions):
      for j in range(total_partitions):
        train = optimizer.minimize(
          loss, var_list=[u_segments[i],u_segments[j]])
        print i,j#,tf_p_submatrix(i,j)
        sess.run(train,feed_dict = {pmi_section:tf_p_submatrix(i,j)})
    print "x after",sess.run(u_segments[i])

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
   a tensorflow helper function used to only compute certain gradients. 
   
   source - https://github.com/tensorflow/tensorflow/issues/9162
-----------------------------------------------------------------------------'''
def entry_stop_gradients(target, mask):
  mask_h = tf.logical_not(mask)

  mask = tf.cast(mask, dtype=target.dtype)
  mask_h = tf.cast(mask_h, dtype=target.dtype)

  return tf.stop_gradient(mask_h * target) + mask * target

'''-----------------------------------------------------------------------------
    t_svd(a)
      this function takes in a 3rd order tensor and computes the t-svd 
      algorithm 
-----------------------------------------------------------------------------'''
def t_svd(a,k):
  max_cores = 20
  n = a[0].shape[0]
 T= len(a)

  a = rotate_tensor(a)

  #shared array must be float64, will be cast to complex128 in processes
  fft_p = mp.rawarray(c_double,2*n*n*(1 + ceil((t-1)/2)))

  #set up cores to compute the fft along 3rd mode
  jobs = []
  core_count = psutil.cpu_count(False)
  process_count = min(core_count,max_cores)

  slices_per_process = n / process_count

  for i in xrange(process_count):
    start = i*slices_per_process
    end = min((i+1)*slices_per_process, n)
    p = mp.process(target=compute_fft, name=i + 1,
                   args=(a[start:end],fft_p,))
    jobs.append(p)
    p.start()

  #wait for processes to finish running
  for p in jobs:
    p.join()

  #start new set of processes to compute each of the symmetric embeddings
  jobs = []

'''-----------------------------------------------------------------------------
    flattened_svd(a)
      this function takes in a tensor in the form of a list of sparse dok 
      matrices, flattens them along their first mode and then computes the top 
      k left singular vectors and uses this for an embedding and an 
      approximation of the core tensor by multiplying each slice from the 
      right with u^t. 
    input:
      a - (list of sparse dok matrices) 
        the tensor representation of the day
      k - (int)
        the number of singular vectors to compute
      save_results - (optional bool)
        whether or not to save the embeddings
    return:
      u - (n x k ndarray)
        the shared embedding in question
      b - (t x k x k ndarray)
        the approximated core tensor
-----------------------------------------------------------------------------'''
def flattened_svd(A,k,save_results = False, parallel = False):
  A_1 = flatten(A)
  T = len(A)
  U, sigma, _ = svds(A_1, k=k, return_singular_vectors="u")

  if save_results:
    np.save("flattened_svd/flattenedsvdU.npy", U)
    np.save("flattened_svd/flattenedsvdsigma.npy", sigma)

  b = np.ndarray((T, k, k))
  if parallel:
    #compute the core tensor in parallel
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    for t in range(T):
      p = mp.Process(target=slice_multiply, args=(A[t], U, t, return_dict))
      jobs.append(p)
      p.start()

    for i in range(T):
      jobs[i].join()


    for t in range(T):
      b[t] = return_dict[t]
  else:
    for t in range(T):
      A[t] = A[t].tocsr()
      b[t] = np.dot(U.T,A[t].dot(U))

  if save_results:
    np.save("flattened_svd/flattenedsvdb",b)


  return U,b

'''-----------------------------------------------------------------------------
    flatten(a)
      this function takes in a list of sparse dok scipy matrices and returns 
      the mode 1 flattened tensor in a csr format. 
    inputs:
      a - (list of sparse dok matrices)
        the tensor representation of the data
    returns:
      a_1 - (csr sparse matrix)
        the mode 1 flattening of the tensor a
    
-----------------------------------------------------------------------------'''
def flatten(a):
  T = len(a)
  n = a[0].shape[0]
  a_1 = sp.dok_matrix((n,T*n))

  for t in range(T):
    for ((i,j), nnz) in a[t].iteritems():
      a_1[i,j + n*t] = nnz

  return a_1.tocsr()

'''-----------------------------------------------------------------------------
    rotate_tensor(a)
      this function takes in a list of n x n sparse matrices representing a 
      n x n x k tensor and returns a list of n x k sparse matrices which 
      represent a n x k x n tensor
    input:
      a - a list of (n x n) sparse dok matrices  
    note:
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

