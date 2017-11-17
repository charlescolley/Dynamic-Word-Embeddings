import numpy as np
from math import sqrt

def main():
  n = 200
  d = 100

  P = np.random.rand(n,n)
  U = np.random.rand(n,d)
  true_result = np.matmul(P, U) - np.matmul(U, np.matmul(U.T, U))
  result = sym_embedding_grad(U.flatten(), P.flatten())
  print sum(result - true_result.flatten())

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


if __name__ == "__main__":
  main()