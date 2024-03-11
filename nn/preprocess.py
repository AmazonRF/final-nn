# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random
from sklearn.utils import shuffle

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    positive_seqs = [seq for seq, label in zip(seqs, labels) if label]
    negative_seqs = [seq for seq, label in zip(seqs, labels) if not label]

    # Determine the class with fewer samples
    flag = -1
    if len(positive_seqs) < len(negative_seqs):
        minority_class = positive_seqs
        majority_class = negative_seqs
        flag = 1
    else: 
        minority_class = negative_seqs
        majority_class = positive_seqs
        flag = 0

    majority_class = negative_seqs if minority_class == positive_seqs else positive_seqs

    # Sample with replacement from the minority class to match the number of samples in the majority class
    sampled_majority_seqs = random.choices(majority_class, k=len(minority_class))


    # Combine the majority class with the upsampled minority class
    sampled_seqs = np.vstack((minority_class, sampled_majority_seqs))

    if flag == 1:
        pos_labels = [1] * len(minority_class)
        neg_labels = [0] * len(sampled_majority_seqs)
        labels = (pos_labels+neg_labels)

    elif flag == 0:
        pos_labels = [1] * len(sampled_majority_seqs)
        neg_labels = [0] * len(minority_class)
        labels = neg_labels+pos_labels


    # Pair each sequence with its label
    paired = list(zip(sampled_seqs, labels))

    # Shuffle to mix while maintaining matches
    shuffled_paired = shuffle(paired)

    # Unzip to separate sequences and labels again if needed
    sampled_seqs, sampled_labels = zip(*shuffled_paired)
    
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    encoded_seqs = []
    
    for seq in seq_arr:
        encoded_seq = [mapping[nuc] for nuc in seq]
        # Flatten the encoded sequence and append to the list
        encoded_seqs.append([item for sublist in encoded_seq for item in sublist])
        
    return np.array(encoded_seqs)