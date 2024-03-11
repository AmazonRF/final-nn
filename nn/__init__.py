"""
BMI203: Biocomputing Algorithms - Winter 2023
Final project: neural networks
"""
from .nn import NeuralNetwork
from .io import (read_text_file,read_fasta_file,sample_window)
from .preprocess import (sample_seqs, one_hot_encode_seqs)
__version__ = "0.1.0"