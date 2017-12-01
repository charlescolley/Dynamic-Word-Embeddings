import pytest
import word2vec as w2v
import process_data as pd
import gradients as grad
import numpy as np

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






def normalize_wordIDs_test()
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