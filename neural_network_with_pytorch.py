import torch
import numpy as np
from torch.autograd import Variable

X = torch.tensor([[1,0,1,0],[1,0,1,1],[0,1,0,1]],dtype=torch.float32)
y = y = torch.Tensor([[1],[1],[0]])

epoch = 5000
lr = 0.1
inputlayer_neurons = X.shape[1] # number of input features 
hiddenlayer_neurons = 3
output_neurons = 1

w0 = Variable(torch.randn(inputlayer_neurons,hiddenlayer_neurons),requires_grad=True).float() # size = (4,3)
b0 = Variable(torch.randn(1, hiddenlayer_neurons),requires_grad=True).float() # size = (1,3)
w1 = Variable(torch.randn(hiddenlayer_neurons, output_neurons),requires_grad=True).float() # size = (3,1)
b1 = Variable(torch.randn(1, output_neurons),requires_grad=True).float() # size = (1,1)

def sigmoid(x):
  return 1/(1+torch.exp(-x))

cost=[]

for i in range(epoch):
  z0 = torch.mm(X,w0) + b0
  a0 = sigmoid(z0)

  z1 = torch.mm(a0, w1) + b1  
  a1 = sigmoid(z1)

  C = ((a1-y)**2).mean()
  cost.append(C)
  w0.retain_grad()
  b0.retain_grad()
  w1.retain_grad()
  b1.retain_grad()
  C.backward()
  if(i%100==0):
    print("After {0} iterations, Loss is {1}".format(i,C))
  w1 = w1-((w1.grad)*lr)
  b1 = b1- ((b1.grad)*lr)

  w0 = w0-((w0.grad)*lr)
  b0 = b0- ((b0.grad)*lr)

import matplotlib.pyplot as plt
plt.plot(range(epoch),cost,'b',label ='Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('actual :\n', y, '\n')
print('predicted :\n', a1)