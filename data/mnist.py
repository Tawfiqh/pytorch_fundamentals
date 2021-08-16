# - name= Finally, a non-toy dataset
#   description= |
#     - Check out the docs for torchvision.datasets ✅
#       - What are the following keyword arguments for? Which are required?
#         - root, train, download ✅
#           root is where to save it -
#           train is  If True, creates dataset from training.pt, otherwise from test.pt.
#           download - downloads the files if needed

#     - Download the MNIST dataset using the relevant pytorch class from pytorch ✅
#     - How long is this dataset?  - 60,000 ✅
#       - What about if train=False when you load it in? - 10,000  ✅
#       - Why do you think there are predetermined train-test splits? Hint= MNIST used to be for benchmarking research  ✅ So that it's predictable and comparable i.e across benchmarks
#     - Take a look at the type of labels this dataset has. What kind of problem is this?
#     - Use the relevant pytorch layers and loss function to create a neural network class capable of representing this problem
#     - Train a neural network on this dataset and evaluate the performance
#     - Compare the performance to a linear model from sklearn (or pytorch if you’d rather) (edited)

import torch
import torchvision

test_mnist = torchvision.datasets.MNIST(root="./data", train=False, download=True)
print(test_mnist[0])
print(test_mnist[0][0].show())
