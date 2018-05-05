import numpy as np

# A layer handles one dimensional input x, apply its computation and return 1-D result y
class Layer:
    # apply layer computation
    def forward(self, x):
        pass

    #  returns dL/dx
    #  saves gradient
    def backward(self, dy):
        pass

    # adjust weight by gradient descend
    def adjust_w(self, lr):
        pass

class dense_layer(Layer):
    def __init__(self, inputNum, unitNum):
        self.w = np.random.uniform(0., 0.1, (inputNum, unitNum))
        self.b = np.random.uniform(0., 0.1, (1, unitNum))
        self.input = None
        self.gradient = None
        self.b_grad =None

    def forward(self, x):
        self.input = x
        return np.dot (x ,self.w)+self.b

    def backward(self, dy):
        self.gradient = (dy*self.input).T
        self.b_grad = dy.T
        return np.dot(self.w , dy)

    def adjust_w(self, lr):
        self.w -= lr*self.gradient
        self.b -= lr*self.b_grad

class relu_layer(Layer):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        y = x.copy()
        y[y<0]=0
        return y

    def backward(self, dy):
        y = self.input.copy()
        y[y>=0] = 1
        y[y<0] = 0
        return dy*y.T

class softmax_layer(Layer):
    def __init__(self):
        self.input = None
        self.output= None

    def forward(self, x):
        self.input = x
        y = np.exp(x)
        self.output = y/np.sum(y)
        return self.output

    def backward(self, dy):
        res = np.dot((np.eye(dy.shape[0]) - self.output * self.output.T), dy)
        return np.dot((np.eye(dy.shape[0]) - self.output*self.output.T),dy)


