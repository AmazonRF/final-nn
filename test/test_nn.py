# TODO: import dependencies and write unit tests below
from nn import NeuralNetwork, one_hot_encode_seqs, sample_seqs
import numpy as np


nn_arch = [
    {'input_dim': 3, 'output_dim': 5, 'activation': 'relu'},
    {'input_dim': 5, 'output_dim': 1, 'activation': 'sigmoid'}  # Assuming a sigmoid output layer for multi-class classification
]
# Instantiate the neural network
nn = NeuralNetwork(nn_arch, lr=0.05, seed=57, batch_size=32, epochs=3000, loss_function='binary_cross_entropy')

def test_single_forward():
    # Generate fake data as shown previously
    W_curr = np.random.rand(3, 5)
    b_curr = np.random.rand(3, 1)
    A_prev = np.random.rand(5, 1)

    #test relu
    activation = "relu"

    cor_Z_curr = np.dot(W_curr, A_prev) + b_curr
    cor_A_curr = NeuralNetwork._relu(nn, cor_Z_curr)
    A_curr, Z_curr = NeuralNetwork._single_forward(nn, W_curr, b_curr, A_prev, activation)

    # Perform assertions here, e.g., check shapes and activation correctness
    assert A_curr.shape == (3, 1), "A_curr shape mismatch"
    assert Z_curr.shape == (3, 1), "Z_curr shape mismatch"
    assert cor_Z_curr.all() == Z_curr.all(), "Z_curr value wrong"
    assert cor_A_curr.all() == A_curr.all(), "A_curr value wrong"

    #test sigmoid
    activation = "sigmoid"

    cor_Z_curr = np.dot(W_curr, A_prev) + b_curr
    cor_A_curr = NeuralNetwork._sigmoid(nn, cor_Z_curr)
    A_curr, Z_curr = NeuralNetwork._single_forward(nn, W_curr, b_curr, A_prev, activation)

    # Perform assertions here, e.g., check shapes and activation correctness
    assert A_curr.shape == (3, 1), "A_curr shape mismatch"
    assert Z_curr.shape == (3, 1), "Z_curr shape mismatch"
    assert cor_Z_curr.all() == Z_curr.all(), "Z_curr value wrong"
    assert cor_A_curr.all() == A_curr.all(), "A_curr value wrong"


def test_forward():

    X = np.array([0,0,0])  # Example input

    # Run the forward method
    actual_output, cache = nn.forward(X)

    # Assertions to check if the actual output matches the expected output
    assert actual_output[0][0] == actual_output[0][1], "If input is all same, the output should all same"
    assert actual_output[0].shape[0] == 5, "the out put shape should be 5"


def test_single_backprop():
    # test data
    W_curr = np.random.rand(3, 5)
    b_curr = np.random.rand(3, 1)
    A_prev = np.random.rand(5, 1)
    y = np.array([1,1,1])
    y_hat = np.array([1,1,0])
    A_curr, Z_curr = NeuralNetwork._single_forward(nn,W_curr, b_curr, A_prev, 'relu')
    dA_curr = np.array([[0.1],[0.1],[0.1]])
    res = NeuralNetwork._single_backprop(nn, W_curr, b_curr,Z_curr,A_prev,dA_curr,'relu') 

    # the output layer should has same dimension as setting
    assert res[0].shape == (5, 1)
    assert res[0].shape == (3, 5)
    assert res[0].shape == (3, 1)


def test_predict():
    X = np.array([0,0,0])  # Example input
    out_y_hat, _ = nn.forward(X)
    predict_y_hat = NeuralNetwork.predict(nn, X)
    assert  predict_y_hat.all() == out_y_hat.all()

def test_binary_cross_entropy():

    #the entropy should be 0 if two array is same
    bin_cross_entropy = NeuralNetwork._binary_cross_entropy(nn,np.array([1,1,1,1]),np.array([1,1,1,1]))

    assert np.allclose(bin_cross_entropy,0, atol=1e-6)


def test_binary_cross_entropy_backprop():
    binary_cross_entropy_backprop_res = NeuralNetwork._binary_cross_entropy_backprop(nn,np.array([1,1,1,20]),np.array([1,1,1,1]))
    assert np.allclose(binary_cross_entropy_backprop_res[0],-1,atol=0.001)
    assert np.allclose(binary_cross_entropy_backprop_res[1],-1,atol=0.001)
    assert np.allclose(binary_cross_entropy_backprop_res[2],-1,atol=0.001)
    assert binary_cross_entropy_backprop_res[3] <0


def test_mean_squared_error():
    assert NeuralNetwork._mean_squared_error(nn,np.array([1,1,1,1]),np.array([1,1,1,0])) == 0.25

def test_mean_squared_error_backprop():
    assert NeuralNetwork._mean_squared_error_backprop(nn,np.array([1,1,1,1]),np.array([1,1,1,0])).all() == np.array([0. , 0. , 0. , 0.5]).all()

def test_sample_seqs():
    X_train = [[1,1,1],[1,0,1],[1,1,0],[0,1,1],[1,1,1],[1,0,1],[1,1,0],[0,1,1]]
    y_train = [1,1,1,0,1,1,1,0]

    new_X_train, new_y_train = sample_seqs(list(X_train), list(y_train))

    assert new_y_train.count(0) == new_y_train.count(1)

def test_one_hot_encode_seqs():

    assert one_hot_encode_seqs(list('ATCG')).all() == np.array([[1, 0, 0, 0],
                                                                [0, 1, 0, 0],
                                                                [0, 0, 1, 0],
                                                                [0, 0, 0, 1]]).all()