import numpy as np
from math import sqrt
from scipy.special import expit
import tensorflow as tf
from numpy.linalg import lstsq


def main():
  tf_least_squares()
  '''
  n = 200
  d = 100

  P = np.random.rand(n,n)
  U = np.random.rand(n,d)
  true_result = np.matmul(P, U) - np.matmul(U, np.matmul(U.T, U))
  result = sym_embedding_grad(U.flatten(), P.flatten())
  print sum(result - true_result.flatten())
  '''

'''
   compute_matvec_from_vectorized_input(array,j,t,n,d,T)
     This function computes the matvec of the j_th word in the shared 
     embedding, with the t_th frontal slice of the core tensor. The 
     dimensions of the n x d shared embedding, and the T x d x d core tensor 
     are passed in to aid in computation. This is a helper function for the 
     gradient function produced by word_to_vec_core_gradient.
   Input:
     array - (array of floats)
       the array with the vectorized shared embedding and core tensor
     j - (int)
       the index of the vector to compute the matvec of.
     t - (int)
       the index of the frontal slice of compute the matvec with. 
     n - (int)
       the dimension (unique word count) of the shared embedding.
     d - (int)
       the dimension of the embedding. 
   Returns:
     x - (numpy array)
       the result of the matvec. 
'''
def compute_matvec_from_vectorized_input(array,j,t,n,d):
  x = np.empty(d)
  for i in range(d):

    x[i] = array[n*d + t*d*d + i]*array[j*d]
    #do row multiplication
    for k in range(1,d):
      x[i] += array[n*d + t*d*d + k*d + i]*array[j*d + k]


'''-----------------------------------------------------------------------------
   compute_rank_1_update_from_vectorized_input(array,t,i,j,alpha,n,d)
       This function is a helper function for the gradient computation for the 
     word_to_vector objective function. This function takes in the shared array,
     the indices for the associated vectors and frontal slice, and computes a 
     rank 1 update to the frontal slice with the ith and jth word embeddings 
     with a scaling coefficient. 
   Input:
     array - (array of floats)
       the array with the vectorized shared embedding and core tensor.
     t - (int)
       the index of the frontal slice to be updated
     i - (int)
       the index of the first word embedding vector to use in the rank1 update
     j - (int)
       the index of the second word embedding vector to use in the rank1 update
     alpha - (float)
       a scaling coefficient to scale the rank 1 update by. 
     n - (int)
       the dimension of the initial word count (unique word count)
     d - (int)
       the dimension of the embedding being produced.
-----------------------------------------------------------------------------'''
def compute_rank_1_update_from_vectorized_input(array,t,i,j,alpha,n,d):
  for k in range(d):
    for l in range(d):
      array[n*d + t*d*d +k*d + l] += alpha*array[i*d + l]*array[j*d + k]


'''-----------------------------------------------------------------------------
   word_to_vec_core_gradient()
       This function takes in a list of co-occurance matrices and a 
     dictionary with the word counts, a dimension, and a negative sampling 
     rate and returns a function which will compute the gradient for the loss 
     function. The objectives being learned are the shared d dimensional 
     embedding and each of the frontal slices of the d x d x T core tensor. 
   Input:
     cooccurrences - (list of sparse dok matrices) 
       The co occurences of words in the text corpus
     word_counts - (list of dictionaries)
       a list of dictionaries linking the index of a given word in a given 
       time slice to the number of times the word shows up in that time 
       slice. for each of the t dictionaries keys are the indexes, and the 
       values are the number of times the word corersponding to that index shows
       up in the t_th corpus.  
     k - (int)
       the negative sampling rate to use
     d - (int)
       the dimension of the embedding to be learned. 
   Returns:
     gradient - (function which takes in an M length array 
                 and returns an M length array)
       scipy.optimize takes in vectorized functions, so though the 
       optimization function will find a n x d array and T, d x d frontal 
       slices of the core tensor, everything will be vectorized. The first nd 
       elements of the input will correspond to the shared embedding, and the 
       next t sets of dd elements will correspond to each frontal slice of 
       the core tensor.  
-----------------------------------------------------------------------------'''
def word_to_vec_core_gradient(cooccurences, word_counts, k, d):

   T = len(cooccurences)
   n = cooccurences[0].shape[0]

   total_counts = np.zeros(T)
   #compute all the total word counts for each time slice for normalization
   for t in range(T):
     for val in word_counts[t].itervalues():
       total_counts[t] += val


   def gradient(vectorized_input):

     #initialize empty array
     output = np.zeros(n*d + T*d*d)

     #compute shared embedding gradient
     for i in range(n):
       for t in range(T):

         B_tw_i = compute_matvec_from_vectorized_input(vectorized_input,i,t,n,d)

         scaling_factor = 0
         #compute j \neq i
         for j in range(i):
           scaling_factor = cooccurences[t][i,j]
           scaling_factor -=  \
           ((k * word_counts[t][i] * word_counts[t][j])/total_counts[t] + \
           cooccurences[t][i,j])*expit(np.dot(vectorized_input[j*d:(j+1)*d],B_tw_i))

           output[d*i:(i+1)*d] += \
             scaling_factor*compute_matvec_from_vectorized_input(
               vectorized_input,j,t,n,d)

         for j in range(i+1,n):
           scaling_factor = cooccurences[t][i, j]
           scaling_factor -= \
             ((k * word_counts[t][i] * word_counts[t][j]) / total_counts[t] + \
              cooccurences[t][i, j]) * expit(
               np.dot(vectorized_input[j * d:(j + 1) * d], B_tw_i))

           output[d * i:(i + 1) * d] += \
             scaling_factor * compute_matvec_from_vectorized_input(
               vectorized_input,j,t,n,d)

         #compute j == i
           scaling_factor = cooccurences[t][i,i]
           scaling_factor -= \
             ((k * word_counts[t][i]**2 / total_counts[t]) + \
              cooccurences[t][i, j]) * expit(
               np.dot(vectorized_input[i * d:(i + 1) * d], B_tw_i))

           output[d * i:(i + 1) * d] += scaling_factor * B_tw_i

     #compute the frontal slices
     for t in range(T):
       for i in range(n):
         B_tw_i = compute_matvec_from_vectorized_input(vectorized_input,i,t,n,d)
         for j in range(n):
           scaling_factor = cooccurences[t][i,j]
           scaling_factor -= \
             ((k * word_counts[t][i]**2 / total_counts[t]) + \
              cooccurences[t][i, j]) * expit(
               np.dot(vectorized_input[j * d:(j + 1) * d], B_tw_i))

           compute_rank_1_update_from_vectorized_input(vectorized_input,t,i,
                                                       j,scaling_factor,n,d)
           














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
      Gram_V[i ,i] += lambda_1
  UVTV = np.dot(U, Gram_V)

  # temporarily add in the lambda_2 term to P
  if lambda_2 != 0.0:
    for i in range(n):
      P[i ,i] += lambda_2
  PV = P * V
  # remove diagonal terms
  if lambda_2 != 0:
    for i in range(n):
      P[i ,i] -= lambda_2
  return PV + UVTV


'''-----------------------------------------------------------------------------
    sym_embedding_grad(P,U)
      This function returns the gradient of the function 
        \|P - UU^T\|_F^2
      with respect to U. The gradient will be of the form of a flattened 
      dense n x d matrix.
    Inputs:
      U - (nd row major array)
        the embedding trying to be learned from the input matrix P 
      P - (nn row major array)
        the dense matrix to be decomposed into a smaller n x d matrix. May 
        have complex entries. 
-----------------------------------------------------------------------------'''
def sym_embedding_grad(U, P):
  n = int(sqrt(P.shape[0]))
  d = int(U.shape[0]/n)

  print n, d

  #store gradient in row major format
  grad_U = np.empty(n*d)
  UTU_jcol = np.empty(d)

  for j in xrange(d):
    #compute jth column of UTU
    for k in xrange(d):
      UTU_jcol[k] = np.conj(U[k]) * U [j]
      for l in xrange(1,n):
        UTU_jcol[k] += np.conj(U[l * d + k]) * U [l * d + j]

    #compute (PU - UUTU)_ij
    for i in xrange(n):
      PU_ij = P[i * n] * U[j]
      for k in xrange(1,n):
        PU_ij += P[i * n + k] * U[k * d + j]

      UUTU_ij = U[i * d] * UTU_jcol[0]
      for k in xrange(1,d):
        UUTU_ij += U[i * d + k] * UTU_jcol[k]

      grad_U[i*d + j] = PU_ij - UUTU_ij  # watch out for loss of precision

  return grad_U

'''-----------------------------------------------------------------------------
    frob_diff_grad(X,A)
      This function returns the gradient of \| X - A\|^2_F with respect to X.
    Which evaluates to -2(X - A). 
    Inputs:
      X - (nm array)
        a flattened matrix as an array which the objecitive function is being 
        differentiated with respect to it.
      A - (nm array)
        The constant matrix flattened as an array.
    Returns:
      grad_X - (nm matrix)
    Notes:
      currently the matrices have not be specified and should just be made 
      sure to be of the same time so the operations will be defined. 
-----------------------------------------------------------------------------'''
def frob_diff_grad(X,A):
  grad_X = 2*(X - A)
  return grad_X

def tf_least_squares():
  n = 10
  m = 10

  A = np.random.rand(n,m)
  b = np.random.rand(n,1)

#  i = tf.placeholder(dtype=tf.int32)
  A_row = tf.placeholder(dtype=tf.float64)
  b_i = tf.placeholder(dtype=tf.float64)
  x = tf.get_variable("x", initializer=np.ones([m,1]))

  loss_func = tf.square(b_i - tf.tensordot(A_row,x,1))

  optimizer = tf.train.GradientDescentOptimizer(.001)
  train = optimizer.minimize(loss_func)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    shuffling = np.random.choice(range(n),n,replace=False)

    for i in xrange(1000):
      j = i % n#shuffling[i % n]
      sess.run(train,feed_dict={A_row: A[j,:],b_i:b[j]})

    print "A:",A
    print "b:",b
    x_true = lstsq(A,b)[0]
    print "true sol:", x_true
    print "tf sol  :",sess.run(x)
    print "norm of diff: ", np.linalg.norm(x_true - sess.run(x))





if __name__ == "__main__":
  main()