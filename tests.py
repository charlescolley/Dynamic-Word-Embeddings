import pytest
import word2vec as w2v
import process_data as pd
import gradients as grad
import numpy as np
import scipy.sparse as sp
from time import time
from random import shuffle

MACHINE_EPS = 1e-13

class TestClass:

  #generate some random matrices and compare against numpy multiply routine
  #errors around ME = 1e-14
  def test_grammian(self):
    test_count = 0
    #square case
    n = 100
    m = 100
    for i in range(test_count):
      A = np.random.rand(n,m)
      assert np.array(map(lambda x: x < MACHINE_EPS,
                      w2v.grammian(A) - np.dot(A.T,A)
                     )).all()
    #n > m case
    n = 100
    m = 50
    for i in range(test_count):
      A = np.random.rand(n,m)
      assert np.array(map(lambda x: x < MACHINE_EPS,
                          w2v.grammian(A) - np.dot(A.T, A)
                          )).all()

    #n <= m case
    n = 50
    m = 100
    for i in range(test_count):
      A = np.random.rand(n,m)
      assert np.array(map(lambda x: x < MACHINE_EPS,
                          w2v.grammian(A) - np.dot(A.T, A)
                          )).all()

  def test_svd_grad(self):
    test_count = 1
    n = 100
    d = 50

    for i in range(test_count):
     P = np.random.rand(n,n)
     U = np.random.rand(n,d)

     true_result = np.matmul(P,U) - np.matmul(U,np.matmul(U.T,U))
     result = np.reshape(grad.sym_embedding_grad(U.flatten(),
                                                      P.flatten())
                         ,[n,d])

     assert np.array(map(lambda x: abs(x) < MACHINE_EPS,
                         true_result - result)).all()






def normalize_intersection_wordIDs_test():
  A = sp.dok_matrix((5, 5))
  B = sp.dok_matrix((5, 5))

  A_dict = {'e': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
  B_dict = {'d': 0, 'c': 1, 'b': 2, 'a': 3, 'f': 4}

  A_val = 1
  B_val = 25
  for i in range(5):
    for j in range(5):
      A[i, j] = A_val
      B[i, j] = B_val

      A_val += 1
      B_val -= 1

  matrix_list = [A, B]
  dict_list = [A_dict, B_dict]

  print "matrices before"
  for matrix in matrix_list:
    print matrix.todense()

  for dict in dict_list:
    print dict

  final_dict = pd.normalize_wordIDs(matrix_list, dict_list)
  print "matrices after"
  for matrix in matrix_list:
    print matrix.todense()
  print final_dict


def normalize_union_wordIDs_test():
  n = 5
  A = sp.dok_matrix((n,n))
  B = sp.dok_matrix((n,n))
  C = sp.dok_matrix((n,n))

  val = 0
  for t in range(3):
    for i in range(n):
      for j in range(n):
        if t == 0:
          A[i,j] = val
          val += 1
        elif t == 1:
          B[i,j] = val
          val += 1
        else:
          C[i,j] = val
          val += 1

  dict_a = {'apple':0,'cat':1,'sandwich':2,'anger':3,"selfie":4}
  dict_b = {'cat':0,'max':1,'chuckle':2,'anger':3,'soda':4}
  dict_c = {'cat':0,'sandwich':1,'charlie':2,'soda':3,'selfie':4}
  P = [A,B,C]
  IDs = w2v.normalize_wordIDs(P,[dict_a,dict_b,dict_c])

  print IDs

  for slice in P:
    print slice.todense()

def permute_dok_test():
  n = 1000
  p =range(n)
  shuffle(p)
  A = sp.rand(n,n,density=.01,format='dok')
  print "starting"
  t = time()
  B = pd.permute_dok_matrix(A,p)
  func_t = time() - t
  print "finished func in {} secs".format(func_t)
  t = time()
  p_A = A[p][:,p]
  array_t = time() - t
  print "finished array access in {} sec".format(array_t)
  print sp.linalg.norm(B - p_A)

def mean_centered_test():
  n = 15
  m = 10

  A = sp.random(n,m,density=.9)

  mean_centered_A = np.dot((np.identity(n) - np.ones((n,n))/n),
                  np.dot(A.todense(),(np.identity(m) - np.ones((m, m))/m)/2))
  mean_centered_AT = np.dot((np.identity(m) - np.ones((m,m))/m),
                  np.dot(A.T.todense(),(np.identity(n) - np.ones((n, n))/n)/2))
  LinOp_A = w2v.mean_center(A)

  x = np.random.rand(m)
  x2= np.random.rand(n)

  A_x = np.dot(mean_centered_A,x)
  AT_x = np.dot(mean_centered_AT,x2)

  LO_A_x = LinOp_A.matvec(x)
  LO_AT_x = LinOp_A.rmatvec(x2)

  print np.linalg.norm(A_x - LO_A_x)
  print np.linalg.norm(AT_x - LO_AT_x)

def matrix_power_test():
  n = 15
  k = 3

  A = sp.random(n, n, density=.7)
  A_k = A.todense()
  for i in range(k-1):
    A_k = np.dot(A_k,A.todense())

  LinOp_A = matrix_power(A,k)

  x = np.random.rand(n)

  LO_A_x = LinOp_A.matvec(x)
  LO_AT_x = LinOp_A.rmatvec(x)

  print np.linalg.norm(np.dot(A_k,x) - LO_A_x)
  print np.linalg.norm(np.dot(A_k.T,x) - LO_AT_x)


def flattened_LO_test():
  n = 15
  m = 10
  T = 2
  dense_A = np.zeros((n, m * T))
  print dense_A.shape
  slices = []
  for t in range(T):
    slices.append(sp.random(n, m, format='dok', density=.5))
    # copy into dense tensor
    print dense_A[:, t * m:(t + 1) * m].shape
    dense_A[:, t * m:((t + 1) * m)] = np.dot(
      (np.identity(n) - np.ones((n, n)) / n),
      np.dot(slices[t].todense(),
             (np.identity(m) - np.ones((m, m)) / m) / 2))

  x = np.random.rand(m * T)
  x2 = np.random.rand(n)

  LO_A = w2v.create_flattened_Linear_Operators(slices, 'mean_center')

  print np.linalg.norm(np.dot(dense_A, x) - LO_A * x)
  print np.linalg.norm(np.dot(dense_A.T, x2) - LO_A.rmatvec(x2))
