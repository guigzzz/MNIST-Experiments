# neural network

basic implementation of feed forward and back propagation algorithms in python using numpy for the linear algebra

- Supports sigmoid, tanh and ReLU for hidden unit activations
- Supports Linear output activation + MSE loss or softmax output activation + cross-entropy loss
- Supports x-fold cross-validation

Usage: python net_wrapper.py

Repository also includes a tensorflow implementation of the same network in tf_net.py.
The tensorflow code is packaged in a python class to make it convenient to use, exposing an interface similar to sklearn's models
