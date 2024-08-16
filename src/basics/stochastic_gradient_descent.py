## The key differences between the implementation of GD and SGD are:
## 1. The weight update is done for each sample in the training set, rather than for the entire training set
## 2. The cost function is calculated for each sample in the training set, rather than for the entire training set
## Efficiency: GD requires the entire dataset to compute the gradients, which can be computationally expensive and slow, especially with large datasets. SGD can be faster because it uses only one data point at a time.
## Convergence Pattern: GD converges smoothly to the minimum if the learning rate is not too high. SGD, on the other hand, shows a lot of variance in the updates to parameters and can oscillate around the minimum, but it can escape local minima more effectively due to its inherent noise.

import numpy as np
import colorama


class CostFunction:
    """Base class for different cost functions"""

    @staticmethod
    def compute(y_true, y_pred):
        pass

    @staticmethod
    def derivative(y_true, y_pred):
        pass


class MSE(CostFunction):
    """Mean Squared Error"""

    @staticmethod
    def compute(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def derivative(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size


class MAE(CostFunction):
    """Mean Absolute Error"""

    @staticmethod
    def compute(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.size


class BinaryCrossEntropy(CostFunction):
    """Binary Cross-Entropy Loss"""

    @staticmethod
    def compute(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.size


class CategoricalCrossEntropy(CostFunction):
    """Categorical Cross-Entropy Loss"""

    @staticmethod
    def compute(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred / y_true.shape[0]


class StochasticGradientDescentOptimizer:
    """Stochastic Gradient Descent Optimizer.
    The batch size is set to 1 for simplicity.
    """

    def __init__(self, cost_function, lr=0.01, iterations=1000):
        self.cost_function = cost_function
        self.lr = lr
        self.iterations = iterations

    def optimize(self, X, y):
        weights = np.random.rand(X.shape[1])
        loss_history = []

        for i in range(self.iterations):
            idx = np.random.randint(len(y))
            X_i = X[idx, :].reshape(1, -1)  # Get the i-th example
            y_i = y[idx]
            predictions = X_i.dot(weights)
            errors = predictions - y_i
            gradient = X_i.T.dot(errors)
            weights -= self.lr * gradient
            cost = self.cost_function.compute(y_i, predictions)
            loss_history.append(cost)

            if i % 100 == 0:
                print(
                    colorama.Fore.GREEN
                    + f"Iteration {i}: Weights = {weights}, Gradients = {gradient}, Loss = {cost}\n"
                )

        return weights, loss_history


class CostFunctionFactory:
    """Factory to create cost function instances"""

    @staticmethod
    def get_cost_function(name):
        cost_functions = {
            "mse": MSE,
            "mae": MAE,
            "binary_crossentropy": BinaryCrossEntropy,
            "categorical_crossentropy": CategoricalCrossEntropy,
        }
        return cost_functions[name]()


## Example usage

if __name__ == "__main__":
    np.random.seed(0)
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([3, 5, 7, 9, 11])

    optimizer = StochasticGradientDescentOptimizer(
        cost_function=CostFunctionFactory.get_cost_function("mse"),
        lr=0.01,
        iterations=1000,
    )
    weights, loss_history = optimizer.optimize(X, y)

    print(f"Final weights: {weights}")
    print(f"Final loss: {loss_history[-1]}")
