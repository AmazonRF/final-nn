# DO NOT MODIFY ANY OF THESE FUNCTIONS
# THEY ARE ALREADY COMPLETE!

# Imports
from typing import List
import random

def read_text_file(filename: str) -> List[str]:
    """
    This function reads a text file as a list of sequences.

    Arguments:
        filename: str
            Filename (should end in .txt).

    Returns:
        seqs: List[str]
            List of sequences.
    """
    with open(filename, "r") as f:
        seqs = [line.strip() for line in f.readlines()]
    return seqs

def read_fasta_file(filename: str) -> List[str]:
    """
    This function reads in a FASTA file as a list of sequence strings.

    Arguments:
        filename: str
            Filename (should end in .fa or .fasta).

    Returns:
        seqs: List[str]
            List of sequences.
    """
    with open(filename, "r") as f:
        seqs = []
        seq = ""
        for line in f:
            if line.startswith(">"):
                seqs.append(seq)
                seq = ""
            else:
                seq += line.strip()
        seqs = seqs[1:]
        return seqs
    

def sample_window(negative_raw_dataset, window_size=17):
    resampleData = []
    for data in negative_raw_dataset:
        start = random.randint(0, len(data) - window_size)
        resampleData.append(data[start:start+window_size])
    return resampleData
