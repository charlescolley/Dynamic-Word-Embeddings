import numpy as np

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
    frob_diff_grad(X,A)
      This function returns the gradient of \| X - A\|^2_F with respect to X.
    Which evaluates to -2(X - A). 
    Inputs:
      X - (n x m matrix)
        The matrix differentiating with respect to 
      A - (n x m matrix)
        The constant matrix.
    Returns:
      grad_X - (n x m matrix)
    Notes:
      currently the matrices have not be specified and should just be made 
      sure to be of the same time so the operations will be defined. 
-----------------------------------------------------------------------------'''
def frob_diff_grad(X,A):
  grad_X = 2(X - A)
  return grad_X