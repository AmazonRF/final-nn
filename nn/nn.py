# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]], # type: ignore
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

        if self._loss_func == 'binary_cross_entropy':
            self.compute_loss = self._binary_cross_entropy
            self.compute_loss_derivative = self._binary_cross_entropy_backprop
        elif self._loss_func == 'mean_squared_error':
            self.compute_loss = self._mean_squared_error
            self.compute_loss_derivative = self._mean_squared_error_backprop


    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # Linear transformation
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        
        # Activation
        if activation == "relu":
            A_curr = self._relu(Z_curr)
        elif activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)  # Assuming you have a similar method for sigmoid
        else:
            raise Exception("Unsupported activation function: " + activation)
        
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        A_curr = X  # Initialize A_curr with the input
        cache = {'A0': X}  # Cache for storing all As and Zs, initialized with input

        # Iterate through each layer
        for idx, layer in enumerate(self.arch, start=1):
            W_curr = self._param_dict['W' + str(idx)]
            b_curr = self._param_dict['b' + str(idx)]
            activation = layer['activation']
            
            # Perform single forward pass for the current layer
            A_prev = A_curr  # Previous layer's activation becomes the current one's input
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)
            
            # Store the current A and Z in cache
            cache['A' + str(idx)] = A_curr
            cache['Z' + str(idx)] = Z_curr

        return A_curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            raise Exception("Unsupported activation function: " + activation_curr)
        
        m = A_prev.shape[1]
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr
        

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}
        try:
            m = y.shape[1]  # Number of examples
        except:
            m = y.shape[0]

        # Calculate the initial gradient of the loss function with respect to the activation
        # This will vary depending on the loss function used
        if self._loss_func == 'binary_cross_entropy':
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mean_squared_error':
            dA_prev = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError("Unsupported loss function.")

        # Iterate through layers in reverse for backpropagation
        for layer_idx in reversed(range(len(self.arch))):
            layer = layer_idx + 1
            dA_curr = dA_prev
            
            A_prev = cache['A' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer)]
            W_curr = self._param_dict['W' + str(layer)]
            b_curr = self._param_dict['b' + str(layer)]
            
            activation_func = self.arch[layer_idx]['activation']
            if activation_func == 'relu':
                dZ_curr = self._relu_backprop(dA_curr, Z_curr)
            elif activation_func == 'sigmoid':
                dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
            else:
                raise ValueError(f"Unsupported activation function: {activation_func}")

            dW_curr = np.dot(dZ_curr, A_prev.T) / m
            db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
            dA_prev = np.dot(W_curr.T, dZ_curr)
            
            grad_dict['dW' + str(layer)] = dW_curr
            grad_dict['db' + str(layer)] = db_curr

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for layer_idx, layer in enumerate(self.arch, start=1):
            # Construct the keys for accessing weights and biases in _param_dict
            W_key = f'W{layer_idx}'
            b_key = f'b{layer_idx}'

            # Construct the keys for accessing the gradients in grad_dict
            dW_key = f'dW{layer_idx}'
            db_key = f'db{layer_idx}'

            # Ensure the gradient keys exist in grad_dict before attempting to update
            if dW_key in grad_dict and db_key in grad_dict:
                # Update the parameters in _param_dict using the gradients in grad_dict
                self._param_dict[W_key] -= self._lr * grad_dict[dW_key]
                self._param_dict[b_key] -= self._lr * grad_dict[db_key]
            else:
                raise KeyError(f"Gradient {dW_key} or {db_key} not found in grad_dict.")

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        for epoch in range(self._epochs):
            # Forward pass on the training set
            y_hat_train, cache_train = self.forward(X_train)
            train_loss = self.compute_loss(y_train, y_hat_train)
            per_epoch_loss_train.append(train_loss)
            
            # Backpropagation to get gradients
            grad_dict = self.backprop(y_train, y_hat_train, cache_train)
            
            # Update model parameters
            self._update_params(grad_dict)
            
            # Forward pass on the validation set
            y_hat_val, cache_val = self.forward(X_val)
            val_loss = self.compute_loss(y_val, y_hat_val)
            per_epoch_loss_val.append(val_loss)
            
            # Optional: Print epoch number and loss information
            print(f"Epoch {epoch+1}/{self._epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # Forward pass to get predictions
        # Assuming that the forward method returns the final layer activation as its first output
        y_hat, _ = self.forward(X)
        
        # Depending on the task, you might need to post-process y_hat
        # For binary classification, for example, you might convert probabilities to binary outcomes
        # For simplicity, this example returns y_hat directly
        
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        sig = 1 / (1 + np.exp(-Z))
        dZ = dA * sig * (1 - sig)
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.array(dA, copy=True)  # Just converting dA to a correct object type if necessary
        dZ[Z <= 0] = 0  # When Z is less than or equal to 0, the gradient is 0
        # No change needed where Z > 0, dA remains as it is because the gradient of ReLU in that region is 1
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        epsilon = 1e-7  # Small constant
        y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)
        
        try:
            m = y.shape[1]  # Number of examples
        except:
            m = y.shape[0]
            
        loss = -1/m * np.sum(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
        return loss


    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        epsilon = 1e-7  # Small constant
        y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)
        
        dA = - (np.divide(y, y_hat_clipped) - np.divide(1 - y, 1 - y_hat_clipped))
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        m = y.shape[1]
        loss = np.sum((y_hat - y)**2) / m
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = 2 * (y - y_hat) / y.shape[1]
        return dA