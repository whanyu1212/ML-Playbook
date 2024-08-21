# ML-Playbook
A collection of notes and hands-on exercises for ML &amp; DL

## Repository Structure
The repository is organized as follows:

- `src/`: Contains implementations of various machine learning algorithms.
- `notebooks/`: Contains Jupyter notebooks for hands-on applications.
- `data/`: Contains sample datasets used in the examples. Sometimes we can just call datasets from the packages directly

## Implementations of basic algorithms from scratch
| Algorithm                     | Description                                                                 | Link to Code Snippet (Clean and refactored)       | Illustration/Use Case                              |
|-------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| Perceptron                    | A linear classifier that updates its weights based on misclassified examples using a simple threshold function | [perceptron.py](src/basics/perceptron.py)         | [Perceptron Notebook](notebooks/perceptron.ipynb) |
| MLP (Multi-Layer Perceptron)  | A feedforward artificial neural network with one or more hidden layers, using backpropagation for training | [mlp.py](src/basics/mlp.py)                              | [MLP Notebook](notebooks/mlp.ipynb)               |
| Gradient Descent              | An iterative optimization algorithm for finding the local minimum of a differentiable function by taking steps proportional to the negative of the gradient | [gradient_descent.py](src/basics/gradient_descent.py)    | [Gradient Descent Notebook](notebooks/gradient_descent.ipynb) |
| Stochastic Gradient Descent   | A variant of gradient descent where the gradient is estimated from a randomly selected subset (mini-batch) of data, providing faster convergence on large datasets | [stochastic_gradient_descent.py](src/basics/stochastic_gradient_descent.py) | [Stochastic Gradient Descent Notebook](notebooks/stochastic_gradient_descent.ipynb) |
| Adaline                      | An adaptive linear neuron that uses a linear activation function and updates weights based on the mean squared error between the predicted and actual values | [adaline.py](src/basics/adaline.py)                      | [Adaline Notebook](notebooks/adaline.ipynb)       |

## Classical ML for structured data

## Deep Learning 