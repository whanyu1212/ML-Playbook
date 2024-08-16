# We can describe the main idea behind gradient descent as climbing down a hill
# until a local or global loss minimum is reached. In each iteration, we take a
# step in the opposite direction of the gradient, where the step size is determined
# by the value of the learning rate, as well as the slope of the gradient
# (for simplicity, the following figure visualizes this only for a single weight, w).

# The following code snippet shows the implementation of the gradient descent algorithm
# for different cost functions in a factory design pattern manner.
# Technicaly they should be stored in different files, but for simplicity, they are
# stored together here for better understanding.

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


class GradientDescentOptimizer:
    """Gradient Descent Optimizer for minimizing cost functions"""

    def __init__(self, cost_function, lr=0.01, iterations=1000):
        self.cost_function = cost_function
        self.lr = lr
        self.iterations = iterations

    def optimize(self, X, y):
        weights = np.random.rand(X.shape[1])
        loss_history = []

        for i in range(self.iterations):
            predictions = X.dot(weights)
            cost = self.cost_function.compute(y, predictions)
            loss_history.append(cost)
            gradients = self.cost_function.derivative(y, predictions).dot(X)
            weights -= self.lr * gradients

            if i % 100 == 0:
                print(
                    colorama.Fore.GREEN
                    + f"Iteration {i}: Weights = {weights}, Gradients = {gradients}, Loss = {cost}\n"
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


# Example usage
if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])

    cost_function = CostFunctionFactory.get_cost_function("mse")
    optimizer = GradientDescentOptimizer(cost_function, lr=0.01, iterations=1000)
    weights, loss_history = optimizer.optimize(X, y)

    print(f"Optimized weights: {weights}")
    print(f"Final loss: {loss_history[-1]}")
    # print(f"Loss history: {loss_history}")
