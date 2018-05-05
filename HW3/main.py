
from layer import *
import numpy as np
import matplotlib.pyplot as plt
# Model is a list of layers

# compute the result of input_x put into model
def forward_pass(input_x, model):
    for layer in model:
        input_x = layer.forward(input_x)

    return input_x

# compute the result of dy back propagated through the model
# the result is somewhat meaningless, what's important is it calls the backward method in each layer
# that method will compute and save the gradient change of weight (if applicable)
def backward_pass(dy, model):
    for layer in model[::-1]:
        dy = layer.backward(dy)

    return dy

# Calles the adjust weight method in each layer, gradient descend.
def adjust_w(model,lr):
    for layer in model:
        layer.adjust_w(lr)

learning_rate = 0.02
num_iter =1000
test_run_times = 1

# Test run for test_run_times times
for test_time in range(test_run_times):

    # create input
    input_x = np.asarray([[0.5, 0.6, 0.1, 0.25, 0.33, 0.9 , 0.88, 0.76, 0.69, 0.95]], dtype= np.float32)
    # create target
    Target = np.asarray([[1,0,0]], dtype= np.float32)
    # create model
    model = [dense_layer(10, 50), relu_layer(), dense_layer(50, 3), softmax_layer()]
    loss_rec = []

    # training for num_iter iterations
    for i in range(num_iter):
        # forward pass
        y = forward_pass(input_x,model)
        # Loss function = -sum(t log y)
        loss_rec.append( -float(np.dot(np.log(y), Target.T)))
        dy = (y-Target).T
        backward_pass(dy,model)
        adjust_w(model, learning_rate)

    # print the output
    # print (y)

# # print out final parameter
# print ('W1--')
# print (model[0].w)
# print ('b1--')
# print (model[0].b)
# print ('W2--')
# print (model[2].w)
# print ('b2--')
# print (model[2].b)

# # plot loss function
# plt.plot(loss_rec)
# plt.ylabel('Loss')
# plt.show()