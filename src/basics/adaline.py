# Very similar to how perceptron works, but with a different cost function
# The cost function is a continuous function that is differentiable


import numpy as np
import colorama


import numpy as np
import colorama


class AdalineGD:
    """ADAptive LInear NEuron classifier using Gradient Descent."""

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            print(colorama.Fore.CYAN + f"EPOCH: {i + 1}")
            output = self.net_input(X)
            errors = y - output
            print(colorama.Fore.GREEN + f"Errors: {errors}")
            self.w_[1:] += self.eta * X.T.dot(errors)
            print(colorama.Fore.RED + f"X.T.dot(errors): {X.T.dot(errors)}")
            print(colorama.Fore.YELLOW + f"Updated weights: {self.w_[1:]}")
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            print(colorama.Fore.BLUE + f"Cost: {cost}\n\n")
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.5, 1, 0)


class AdalineSGD:
    """ADAptive LInear NEuron classifier using Stochastic Gradient Descent."""

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.0)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


class Adaline:
    """Unified ADAptive LInear NEuron classifier supporting both GD and SGD."""

    def __init__(self, method="gd", **kwargs):
        if method == "gd":
            self.model = AdalineGD(**kwargs)
        elif method == "sgd":
            self.model = AdalineSGD(**kwargs)
        else:
            raise ValueError("Method must be either 'gd' or 'sgd'")

    def fit(self, X, y):
        return self.model.fit(X, y)

    def partial_fit(self, X, y):
        if hasattr(self.model, "partial_fit"):
            return self.model.partial_fit(X, y)
        else:
            raise NotImplementedError("Partial fit is not available for GD")

    def predict(self, X):
        return self.model.predict(X)

    def net_input(self, X):
        return self.model.net_input(X)

    def activation(self, X):
        return self.model.activation(X)


# Example usage

if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([3, 5, 7, 9, 11])

    adaline = Adaline(method="gd", eta=0.01, n_iter=100)
    adaline.fit(X, y)

    adaline = Adaline(method="sgd", eta=0.01, n_iter=100)
    adaline.fit(X, y)
