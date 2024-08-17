## Implementation of Multi-Layer Perceptron (MLP) from scratch without using any deep learning libraries.
import numpy as np


class MLP:
    def __init__(
        self, num_features: np.array, num_hidden: int, num_classes: int, random_seed=123
    ) -> None:
        """Initialize a shallow neural network with one hidden layer which
        uses the sigmoid activation function for the hidden layer and output layer.
        This is also known as a Multi-Layer Perceptron (MLP).

        Args:
            num_features (np.array): The number of features in the input data
            num_hidden (int): The number of neurons in the hidden layer
            num_classes (int): The number of classes in the output layer
            random_seed (int, optional): seed for rgen. Defaults to 123.
        """
        # super().__init__()

        self.num_classes = num_classes

        # hidden
        rng = np.random.default_rng(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    @staticmethod
    def sigmoid(z: np.array) -> np.array:
        """The sigmoid activation function

        Args:
            z (np.array): The net input of the neuron
            that we want to apply the sigmoid function to

        Returns:
            np.array: The output of the neuron after applying the sigmoid function
        """
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def int_to_onehot(y: np.array, num_labels: int) -> np.array:
        """Convert integer labels to one-hot encoded vectors

        Args:
            y (np.array): The integer labels
            num_labels (int): The number of unique labels

        Returns:
            np.array: The one-hot encoded labels stored in a 2D array
        """

        arr = np.zeros((y.shape[0], num_labels))
        for i, val in enumerate(y):
            arr[i, val] = 1

        return arr

    def forward(self, x: np.array) -> tuple[np.array, np.array]:
        """Forward propagation

        Args:
            x (np.array): The input data

        Returns:
            np.array: The output of the network after forward propagation
        """
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = MLP.sigmoid(z_h)

        # Output layer
        # input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = MLP.sigmoid(z_out)
        return a_h, a_out

    def backward(
        self, x: np.array, a_h: np.array, a_out: np.array, y: np.array
    ) -> tuple[np.array, np.array, np.array, np.array]:

        #########################
        ### Output layer weights
        #########################

        # onehot encoding
        y_onehot = MLP.int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use

        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1.0 - a_out)  # sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out  # "delta (rule) placeholder"

        # gradient for output weights

        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h

        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        #################################
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight

        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out

        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1.0 - a_h)  # sigmoid derivative

        # [n_examples, n_features]
        d_z_h__d_w_h = x

        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)
