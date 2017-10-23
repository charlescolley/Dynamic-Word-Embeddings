import pytest
import word2vec as w2v
import numpy as np

MACHINE_EPS = 1e-13

class TestClass:

  #generate some random matrices and compare against numpy multiply routine
  #errors around ME = 1e-14
  def test_grammian(self):
    test_count = 5
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
