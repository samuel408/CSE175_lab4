#
# arch.py
#
# This script implements three Python classes for three different artificial
# neural network architectures: no hidden layer, one hidden layer, and two
# hidden layers. Note that this script requires the installation of the
# PyTorch (torch) Python package.
#
# This content is protected and may not be shared, uploaded, or distributed.
#
# PLACE ANY COMMENTS, INCLUDING ACKNOWLEDGMENTS, HERE.
#I had to use the iris_csv_url from catCourses.
#refered heavily to the PA4 PDF aswell as pytorch and pandas documentation

# ALSO, PROVIDE ANSWERS TO THE FOLLOWING TWO QUESTIONS.
#
# todo: Which network architecture achieves the lowest training set error?
     #Answer: the training model with one hidden layer(AnneOneHid)  has the lowest training set error on epoch 90!

# todo: Which network architecture tends to exhibit the best testing set accuracy?
    # Answer: the training model with one hidden layer,AnnOneHid,has the best training set accuracy!

#
# PLACE YOUR NAME AND THE DATE HERE
#Samuel Saldivar 12/12/2022

# PyTorch - Deep Learning Models
import torch.nn as nn
import torch.nn.functional as F


# Number of input features ...
input_size = 4
# Number of output classes ...
output_size = 3


class AnnLinear(nn.Module):
    """Class describing a linear artificial neural network, with no hidden
    layers, with inputs directly projecting to outputs."""

    def __init__(self):
        super().__init__()
        # PLACE NETWORK ARCHITECTURE CODE HERE
        #todo: intialize networl  with the input and output sizes above.
        self.my_layer = nn.Linear(in_features=input_size, out_features=output_size)


    def forward(self, x):
        # PLACE YOUR FORWARD PASS CODE HERE
        #todo: set up code from PDF
        my_layer_net = self.my_layer(x)
        y_hat = my_layer_net
        return y_hat



class AnnOneHid(nn.Module):
    """Class describing an artificial neural network with one hidden layer,
    using the rectified linear (ReLU) activation function."""

    def __init__(self):
        super().__init__()
        # PLACE NETWORK ARCHITECTURE CODE HERE
        # todo: create network
        # The network architecture with a single hidden layer should use 20 hidden units
        self.my_layer = nn.Linear(in_features=input_size, out_features=20)
        #todo: create hidden layer
        self.my_layerhidden= nn.Linear(in_features=20, out_features=output_size)

    def forward(self, x):
        # PLACE YOUR FORWARD PASS CODE HERE
        my_layer_net = self.my_layer(x)
        my_layer_act = F.relu(my_layer_net)
        y_hat = self.my_layerhidden(my_layer_act)
        return y_hat


class AnnTwoHid(nn.Module):
    """Class describing an artificial neural network with two hidden layers,
    using the rectified linear (ReLU) activation function."""

    def __init__(self):
        super().__init__()
        # PLACE NETWORK ARCHITECTURE CODE HERE
        #todo:The net-work architecture with two hidden layers should have 16 units in the first hidden layer and 12 inthe second.
       #todo:Output layers should compute the linear weighted sum of their inputs, without any onlinear activation function.
        self.my_layer = nn.Linear(in_features=input_size, out_features=16)
        self.my_firstHiddenlayer = nn.Linear(in_features=16, out_features=12)
        self.my_SecondHiddenLayer = nn.Linear(in_features=12, out_features=output_size)
    def forward(self, x):
        # PLACE YOUR FORWARD PASS CODE HERE
        my_layer_net = self.my_layer(x)
        my_layer_act = F.relu(my_layer_net)
        my_layer_net_Hid = self.my_firstHiddenlayer(my_layer_act)
        my_layer_act_Hid = F.relu(my_layer_net_Hid)
        y_hat = my_layer_act_Hid

        return y_hat
