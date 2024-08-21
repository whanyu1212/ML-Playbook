## Implementation of Multi-Layer Perceptron (MLP) from scratch without using any deep learning libraries.
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)


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

        self.num_classes = num_classes

        # hidden
        rng = np.random.default_rng(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    @staticmethod
    def softmax(z: np.array) -> np.array:
        """Apply the softmax activation function to the net input z.

        Args:
            z (np.array): The net input to the output layer

        Returns:
            np.array: activated output from the output layer
        """
        e_z = np.exp(
            z - np.max(z, axis=1, keepdims=True)
        )  # Subtract max for numerical stability
        return e_z / e_z.sum(axis=1, keepdims=True)

    @staticmethod
    def sigmoid(z: np.array) -> np.array:
        """Apply the sigmoid activation function to the net input z.

        Args:
            z (np.array): The net input to the hidden layer

        Returns:
            np.array: activated output from the hidden layer
        """

        # avoid numerical instability
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

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
        """Perform forward propagation through the network.

        Args:
            x (np.array): Input data of shape (num_samples, num_features)

        Returns:
            tuple[np.array, np.array]: Activations from the hidden layer and output layer
        """
        # Compute activations in the hidden layer

        # the net input to the hidden layer, pre-activation
        hidden_input = np.dot(x, self.weight_h.T) + self.bias_h
        # the output from the hidden layer, post-activation
        hidden_output = MLP.sigmoid(hidden_input)

        # Compute the network output
        output_input = np.dot(hidden_output, self.weight_out.T) + self.bias_out
        network_output = MLP.softmax(output_input)

        return hidden_output, network_output

    def backward(
        self,
        x: np.array,
        hidden_output: np.array,
        network_output: np.array,
        y: np.array,
    ) -> tuple[np.array, np.array, np.array, np.array]:
        """Perform backward propagation through the network to compute the gradients.

        Args:
            x (np.array): Input data
            hidden_output (np.array): output from the hidden layer
            network_output (np.array): output from the output layer
            y (np.array): True labels

        Returns:
            tuple[np.array, np.array, np.array, np.array]:
            Gradients for output weights, output biases, hidden weights, and hidden biases
        """
        # One-hot encode y if not already done
        onehot_y = MLP.int_to_onehot(y, self.num_classes)

        # Calculate error at output layer (softmax - onehot encoded labels)
        output_error = (network_output - onehot_y) / x.shape[0]

        # Gradients for output layer
        grad_weight_out = np.dot(output_error.T, hidden_output)
        grad_bias_out = np.sum(output_error, axis=0)

        # Backpropagate the error to the hidden layer
        hidden_error = (
            np.dot(output_error, self.weight_out) * hidden_output * (1 - hidden_output)
        )

        # Gradients for hidden layer
        grad_weight_h = np.dot(hidden_error.T, x)
        grad_bias_h = np.sum(hidden_error, axis=0)

        return grad_weight_out, grad_bias_out, grad_weight_h, grad_bias_h

    def update_weights(
        self,
        gradients: tuple[np.array, np.array, np.array, np.array],
        learning_rate: float,
    ) -> None:
        """Update the weights according to the computed gradients
        from backpropagation using batch gradient descent.

        Args:
            gradients (tuple[np.array, np.array, np.array, np.array]):
            Gradients for output weights, output biases, hidden weights, and hidden biases
            learning_rate (float): Step size at each iteration.
        """
        grad_weight_out, grad_bias_out, grad_weight_h, grad_bias_h = gradients

        # Update the output layer weights and biases
        self.weight_out -= learning_rate * grad_weight_out
        self.bias_out -= learning_rate * grad_bias_out

        # Update the hidden layer weights and biases
        self.weight_h -= learning_rate * grad_weight_h
        self.bias_h -= learning_rate * grad_bias_h

    def fit(
        self,
        X_train: np.array,
        y_train: np.array,
        X_val: np.array,
        y_val: np.array,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> None:
        """Fit the MLP model to the given data by batch gradient descent.

        Args:
            X_train (np.array): Training input data
            y_train (np.array): Training true labels
            X_val (np.array): Validation input data
            y_val (np.array): Validation true labels
            epochs (int): Number of epochs to train the model
            batch_size (int): Number of samples in each mini-batch
            learning_rate (float): Step size to update the weights in each iteration.
        """
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            # Shuffle data at the beginning of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                hidden_output, network_output = self.forward(X_batch)

                # Backward pass
                gradients = self.backward(
                    X_batch, hidden_output, network_output, y_batch
                )
                self.update_weights(gradients, learning_rate)

            train_loss = self.compute_loss(X_train, y_train)
            train_predictions = self.predict(X_train)
            train_accuracy = np.mean(y_train == train_predictions)

            # Validation loss and accuracy
            val_loss = self.compute_loss(X_val, y_val)
            val_predictions = self.predict(X_val)
            val_accuracy = np.mean(y_val == val_predictions)

            print(
                f"Epoch {epoch+1}, "
                f"Train Loss: {Fore.RED}{train_loss:.4f}{Style.RESET_ALL}, "
                f"Train Accuracy: {Fore.GREEN}{train_accuracy:.4f}{Style.RESET_ALL}, "
                f"Val Loss: {Fore.RED}{val_loss:.4f}{Style.RESET_ALL}, "
                f"Val Accuracy: {Fore.GREEN}{val_accuracy:.4f}{Style.RESET_ALL}"
            )

    def compute_loss(self, X: np.array, y: np.array) -> float:
        """Compute the loss of the model on the given data.
        In the case of the MLP, we use the categorical cross-entropy loss.

        Args:
            X (np.array): Input data
            y (np.array): True labels

        Returns:
            float: The loss value
        """
        # we dont need the hidden output here, and thus _
        _, network_output = self.forward(X)
        m = y.shape[0]
        y_onehot = MLP.int_to_onehot(y, self.num_classes)
        log_likelihood = -np.log(network_output + 1e-9)
        # adding a small constant for numerical stability
        loss = (y_onehot * log_likelihood).sum() / m
        return loss

    def predict(self, X: np.array) -> np.array:
        """Predict the class labels for the given data.

        Args:
            X (np.array): input data

        Returns:
            np.array: Predicted class labels
        """

        _, network_output = self.forward(X)
        return np.argmax(network_output, axis=1)

    def evaluate(self, X: np.array, y: np.array) -> float:
        """evaluate the accuracy of the model on the given data.

        Args:
            X (np.array): input data
            y (np.array): true labels

        Returns:
            float: accuracy of the model
        """
        predictions = self.predict(X)
        accuracy = np.mean(y == predictions)
        return accuracy
