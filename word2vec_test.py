import pytest
import word2vec as w2v
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




