import torch
import numpy as np
import matplotlib.pyplot as plt
from evaluate import get_sentences
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def build_vocab():
    # Build a list with the modified vocabulary of "Music with Expressivity..."
    # size: 267+1PAD = 268
    vocab = []
    for note in range(128):
        vocab.append("NOTE_ON<{}>".format(note))
        vocab.append("NOTE_OFF<{}>".format(note))

    for tick in [120, 240, 360, 480, 600, 720, 840, 960]:
        vocab.append("TICKSHIFT<{}>".format(tick))

    for token in ["<START>", "<END>", "SILENCE", "<PAD>"]:
        vocab.append(token)

    return vocab

def encode_song(token_to_int, MAX_SEQ_LENGTH, song):
    #Encode a song into a sequence of one-hots
    """
    Encode a song into a sequence of idx pointing to a token in the vocabulary
    Pad the sequence to a=have S=maximum sequence length in training
    :param token_to_int: dict
    :param MAX_SEQ_LENGTH: int
    :param song: np.array
    :return:
    """
    encoded_song = []

    while len(encoded_song) != (MAX_SEQ_LENGTH-len(song)):
        idx = token_to_int["<PAD>"]
        one_hot = torch.zeros((268,))
        one_hot[idx] = 1
        encoded_song.append(one_hot)

    for word in song:
        idx = token_to_int[word]
        one_hot = torch.zeros((268,))
        one_hot[idx] = 1
        encoded_song.append(one_hot)


    encoded_song = np.stack(encoded_song)

    return encoded_song

def decode_song(int_to_token, encoded_song):
    #Decode a song (np.array)
    #returns a list of strings
    song = []
    for onehot in encoded_song:
        word = int_to_token[np.argmax(onehot)]
        song.append(word)

    return song

def build_training_batch(datapath):
    """
    Create a one hot encoding for all the input data
    """
    corpus = get_sentences(datapath)
    x_lens = [len(x) for x in corpus] #length of sequences before padding

    MAX_SEQ_LENGTH = 0  # 728 when not quantizing more
    for song in corpus:
        seqlength = len(song)
        if seqlength > MAX_SEQ_LENGTH:
            MAX_SEQ_LENGTH = seqlength

    vocab = build_vocab()

    #Transform and Inverse transform of vocabulary tokens
    token_to_int = dict((token, i) for i, token in enumerate(vocab))
    int_to_token = dict((i, token) for i, token in enumerate(vocab))

    train = []
    # Encode the training corpus and pad the sequences
    for song in corpus:
        encoded_song = encode_song(token_to_int, MAX_SEQ_LENGTH, song)
        train.append(encoded_song)


    batch = torch.Tensor(np.stack(train)) #Size: B x S x F == 229 x 728 x 268

    return batch, x_lens, token_to_int, int_to_token


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### Preprocessing
    ##
    datapath = r"C:\Users\isabelle\Documents\PythonScripts\ift6010-h21-team1\corpus.txt"
    batch, x_lens, token_to_int, int_to_token = build_training_batch(datapath)
    #
    # corpus = get_sentences(datapath)
    # #get maximum sequence length
    # VOCAB_LENGTH =  268 # from event definition in quantize + 1 for PAD token
    # MAX_SEQ_LENGTH = 0 #728 when not quantizing more
    # for song in corpus:
    #     seqlength = len(song)
    #     if seqlength > MAX_SEQ_LENGTH:
    #         MAX_SEQ_LENGTH = seqlength
    #
    # #Encode each 'word' into a 267 one-hot vector
    #
    # vocab = build_vocab()
    #
    # #build one-hot
    # token_to_int = dict((token, i) for i, token in enumerate(vocab))
    # int_to_token = dict((i, token) for i, token in enumerate(vocab))
    #
    # train = []
    # for song in corpus:
    #     encoded_song = encode_song(token_to_int, MAX_SEQ_LENGTH, song)
    #     train.append(encoded_song)

    """
    for song in corpus:
        
    
    """
    """
    Notes
    
    2 RNNs
    Encoder input: sequence [wordidx1, wordidx2,...,<end>], out: ctxt single vector
    Decoder in:encoder ctxt vect, out: sequence
    
    task: predict next note given sequence
    teacher forcing: 
    trainingL input sequences, output: first note
    
    1. Create a one-hot encoding of the input and outputs
    
    Problem: song sequences dont have the same length
    Need padding 
    I can use .transpose(dim0,dim1) and torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0.0)
    -> I need to add a pad token in my vocabulary!
    -> Pad at the beginnning so Model learns to predict what's next
    ->Then do the encoding
    
    """