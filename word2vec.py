import theano.tensor as T
import tensorflow as tf
import theano as t
import downhill
import numpy as np

def main():
  downhillExample()

def TensorFlowTutorial():
  vocab_size = 10000
  embedding_size = 100
  embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1,1))


def theanoCode():
  x = t.shared(np.ones(2),'x')
  y = t.shared(np.ones(2), 'y')

  matrix = np.array([[1,2],[3,4]])

  x_1, x_2 = T.dvectors('x_1','x_2')
  A = T.dmatrix('A')
  qf = T.dot(x,T.dot(A,y))
  sigma = 1 / (1 + T.exp(-y))
  logistic = t.function([y],sigma)

  print downhill.minimize(qf, matrix,inputs=A)


def downhillExample():

#  THEANO_FLAGS = None
  m = t.shared(np.ones((1, ), dtype=np.float64), name='m')
  b = t.shared(np.zeros((1, ), dtype=np.float64), name='b')

  x = T.vector('x')
  y = T.vector('y')

  loss = T.sqr(m * x + b - y).sum()
  sizes = np.array([1200,2013,8129,2431,2211])
  prices = np.array([103020, 203310, 3922013, 224321, 449020])

  downhill.adaptive.ADAGRAD(loss, params=[sizes,prices],inputs=[x,y])
  #downhill.minimize(loss,[sizes, prices],inputs = [x,y])

  print m, b


#minimize a quadratic form


#def word2vec_loss_func(pmi):




if __name__ == "__main__":
  main()