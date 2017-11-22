import torch
from torch.autograd import Variable
from torch import optim
import random
import numpy
from numpy import linalg

def main():
  shuchins_pytorch_demo()


def shuchins_pytorch_demo():
  dtype = torch.FloatTensor
  M, N = 100, 50
  # setting up a random problem
  A = Variable(torch.randn(M, N).type(dtype), requires_grad=False)
  x0 = Variable(torch.randn(N, 1).type(dtype), requires_grad=False)
  y = A.mm(x0)

  x = Variable(torch.randn(N, 1).type(dtype), requires_grad=True)
  learning_rate = 0.01
  for t in range(1000):
    i = random.randint(0, M-1)  # picking a random row from A
    a1 = A[i,:]
    a1 = a1.unsqueeze(0)
    y_pred = torch.mm(a1,x)
    loss = (y_pred - y[i]).pow(2)
    print (t,loss.data)
    loss.backward()
    x.data -= learning_rate * x.grad.data
   # learning_rate = 0.001 / (t + 1)
    x.grad.data.zero_()


if __name__ == "__main__":
  main()