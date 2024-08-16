# Implementing a simple Percentron model for binary classification from scratch

# Typically, X and y are passed to the fit method when you are ready to train the model
# It is not a common practice to initialize X and y with the class constructor

import numpy as np
import colorama


class Perceptron:
    """
    Perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Number of passes over the training dataset (epochs).
    random_state : int, optional (default=42)
        Random number generator seed for random weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta: float, n_iter: int, random_state: int = 42) -> None:
        """Initializing the Perceptron class with the learning rate, number of iterations, and random state

        Args:
            eta (float): eta is the learning rate, which is a constant between 0.0 and 1.0
            n_iter (int): n_iter is the number of epochs, which is the number of passes over the training dataset
            random_state (int, optional): State for the rgen. Defaults to 42.
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        # You can use RandomState as well but default_rng is recommended for better randomness
        self.rgen = self.rgen = np.random.default_rng(self.random_state)

    def fit(self, X: np.array, y: np.array) -> "Perceptron":
        # initial weights are drawn from a normal distribution with mean 0 and standard deviation 0.01
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # list to store the number of misclassifications in each epoch
        self.errors_ = []

        for _ in range(self.n_iter):
            print(colorama.Fore.GREEN + f"Epoch {_ + 1}:")
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                print(
                    colorama.Fore.CYAN
                    + f"xi: Updating weights {self.w_[1:]} by {update * xi}"
                )
                self.w_[0] += update
                print(
                    colorama.Fore.CYAN
                    + f"bias: Updating bias {self.w_[0]} by {update}\n"
                )
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self  # returns the perceptron instance itself, allowing method chaining

    def net_input(self, X: np.array) -> np.array:
        """Calculate the net input by taking the dot product of the input and the weights

        Args:
            X (np.array): input array

        Returns:
            np.array: output of the dot product to be used in the predict method
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X: np.array) -> np.array:
        """Predict the class label after the unit step function is applied

        Args:
            X (np.array): input array

        Returns:
            np.array: predicted class labels based on the unit step function
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# Example usage

if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    print(ppn.errors_)
    print(ppn.predict(X))
