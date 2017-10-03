import theano.tensor as T
import tensorflow as tf
import theano as t
import downhill
import numpy as np

def main():
  downhillExample()

'''
    build_objective_functions(word_count_matrix, k)
      This function takes in a n x m matrix with the scaled number of times a 
      word appears within the context c (#(w,c)) and returns a lambda 
      function which computes the loss function and a gradient function. The 
      function will 
    Inputs:
      word_count_matrix - (sparse n x m matrix)
        a scipy sparse matrix which the (i,j)th entry corresponds to #(w_i,
        c_j)|D|. Here |D| denotes the number of words in the text corpus. 
      word_count - (dictionary)
        a dictionary which has the word counts for a text corpus. 
      k - int
        the negative sampling rate, creates k fake samples which help prevent 
        unform distriutions from arising. Samples are created from a unigram 
        distriution. P((w,c)) = #(w)*#(c)^{3/4}/Z where Z is a normalizing 
        constant. 
    Returns:
      loss_func - (lambda func)
        an anonymous function which has the negated word2vec objective 
        function (which is typically maximized). 
        \sum_{(w,c) \in D} (log(softmax(v_c,v_w)) - k 
      gradient - (lambda func)
        an anonymous function which has the gradient of the 
'''
def build_loss_function(word_count_matrix, word_count, k):


def tensorflow_tutorial():
  vocab_size = 10000
  embedding_size = 100
  embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size],-1,1))


def theano_code():
  x = t.shared(np.ones(2),'x')
  y = t.shared(np.ones(2), 'y')

  matrix = np.array([[1,2],[3,4]])

  x_1, x_2 = T.dvectors('x_1','x_2')
  A = T.dmatrix('A')
  qf = T.dot(x,T.dot(A,y))
  sigma = 1 / (1 + T.exp(-y))
  logistic = t.function([y],sigma)

  print downhill.minimize(qf, matrix,inputs=A)


def downhill_example():

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