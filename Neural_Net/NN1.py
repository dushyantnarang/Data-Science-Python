import numpy as np
import matplotlib.pyplot as plt

#input data
x = np.array([[0,1,0],[0,1,1],[0,0,0],[1,0,0],[1,1,1],[1,0,1]])
y = np.array([[0],[0],[0],[1],[1],[1]])

class NeuralNetwork:
    def __init__(self,x,y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)
        self.error_history = []
        self.epoch_list = []

    def sigmoid(self, p):
        return 1/(1 + np.exp(-p))

    def sigmoid_derivative(self, q):
        return q*(1-q)

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input,self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        #application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output)*self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output)*self.sigmoid_derivative(self.output), self.weights2.T)*self.sigmoid_derivative(self.layer1)))
        self.error = self.y - self.output

        #update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self,epochs=25000):
        for epoch in range(epochs):
            self.feedforward()
            self.backprop()
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

NN = NeuralNetwork(x,y)
NN.train()

plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
