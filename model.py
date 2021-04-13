import torch
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

from evaluate import get_sentences
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# From https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

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

class EncoderRNN(nn.Module):
    def __init__(self, in_size, hidden_size, n_stack=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(in_size, hidden_size, n_stack, batch_first=True) #embedding size == hidden_size

    def forward(self, input, hidden):
        out, hidden = self.gru(input, hidden)
        return out, hidden

    def initHidden(self):
        return torch.zeros(1,1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, out_size, n_stack=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_stack = n_stack

        self.gru = nn.GRU(hidden_size, hidden_size, n_stack, batch_first=True)
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        out, hidden = self.gru(input, hidden)
        out = self.softmax(self.out(out[0]))
        return out, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def train(batch, x_lens,token_to_int, encoder, decoder, encoder_optim, decoder_optim, criterion, max_length=728):

    encoder_hidden = encoder.initHidden()

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    loss=0

    #batch Size: B x S x F == 229 x 728 x 268
    seq_len = batch.shape[1]
    for song in batch:
        #Take all the words in the sequence through the Encoder to build the context vector
        for ei in range(seq_len):
            encoder_out, encoder_hidden = encoder(song[ei].unsqueeze(0).unsqueeze(0), encoder_hidden) # (b,seq, in_size) == (1,1,voc)
            # encoder_outs[ei] = encoder[0,0]

        #Build the start 1-hot
        start_one_hot = torch.zeros(batch.shape[2])
        start_one_hot[token_to_int['<START>']] = 1
        start_token = torch.Tensor(start_one_hot).unsqueeze(0).unsqueeze(0)
        decoder_input = start_token.to(device=device)

        decoder_hidden = encoder_hidden #1,1,268

        #teacher forcing
        for di in range(seq_len):
            decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss = loss + criterion(decoder_out, song[di].unsqueeze(0).unsqueeze(0)) #Criterion is not working!
            decoder_input = song[di] #give ground truth to the model

    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.item()/ seq_len

def trainIters(encoder, decoder, batch, x_lens, token_to_int, n_iter, print_every=1000, plot_every=100, lr=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0 #reset every print_every
    plot_loss_total = 0 #reset every plot_every

    encoder_optim = optim.SGD(encoder.parameters(), lr)
    decoder_optim = optim.SGD(decoder.parameters(), lr)

    criterion = nn.NLLLoss()

    for iter in range(1,n_iter+1):
        loss = train(batch, x_lens, token_to_int, encoder, decoder, encoder_optim, decoder_optim, criterion, max_length=728)

        print_loss_total = print_loss_total + loss
        plot_loss_total = plot_loss_total + loss

        if iter % print_every ==0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iter),
                                         iter, iter / n_iter * 100,
                                         print_loss_avg))

        if iter % plot_every ==0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

    return 0
if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### Preprocessing
    ##
    datapath = r"C:\Users\isabelle\Documents\PythonScripts\ift6010-h21-team1\corpus.txt"

    batch, x_lens, token_to_int, int_to_token = build_training_batch(datapath)
    #Technically no need to pack sequences...hum...
    """
    batch_packed = pack_padded_sequence(batch, x_lens, batch_first=True, enforce_sorted=False)

    embedding_dim, h_dim, n_layers = batch.shape[2], 2, 3
    x_packed = batch_packed
    rnn = nn.GRU(embedding_dim, h_dim, n_layers, batch_first=True)
    # output_packed, hidden = rnn(x_packed, hidden)
    output_packed, hidden = rnn(x_packed)

    #unpack output
    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output_packed) #Shape 728, 229, 268
    """
    enc_hidden_size = 100
    encoder = EncoderRNN(in_size=batch.shape[2],hidden_size=batch.shape[2],n_stack=1).to(device)
    decoder = DecoderRNN(hidden_size=batch.shape[2], out_size=batch.shape[2],n_stack=1).to(device)

    #train
    n_iter = 10
    trainIters(encoder, decoder, batch, x_lens, token_to_int, n_iter, print_every=1000, plot_every=100, lr=0.01)
    print("Dodeee!")
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