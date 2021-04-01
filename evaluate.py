from nltk import bigrams, trigrams
from collections import Counter, defaultdict
from mido import Message, MidiFile, MidiTrack
from datetime import datetime
import re
import random
import numpy as np

def get_sentences(datapath):
    # Get a list of tokenized sentences from a corpus
    data = []
    f = open(datapath, "r")
    for line in f:
        sent = [word for word in line.split(' ') if word is not '\n']
        data.append(sent)
    f.close()

    return data

def get_ngram_model(data):
    # Create a placeholder for model
    # Code adapted from https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
    # ?utm_source=blog&utm_medium=Natural_Language_Generation_System_using_PyTorch
    model = defaultdict(lambda: defaultdict(lambda: 0))

    # Count frequency of co-occurance
    for sentence in data:
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

    # Let's transform the counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count

    return model

def generate_song(model, text=["<START>", "NOTE_ON<72>"], maxLength=30):
    # Code adapted from https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-language-model-nlp-python-code/
    # ?utm_source=blog&utm_medium=Natural_Language_Generation_System_using_PyTorch
    sentence_finished = False
    while not sentence_finished:
        # select a random probability threshold
        # r = random.random()
        # accumulator = .0
        #
        # for word in model[tuple(text[-2:])].keys():
        #     accumulator += model[tuple(text[-2:])][word]
        #     # select words that are above the probability threshold
        #     if accumulator >= r:
        #         text.append(word)
        #         break
        if text[-2:] == [None, None]:
            sentence_finished = True

        elif len(text) == maxLength:
            text.append("<END>")
            break

        else:
            # Select the word with the highest probability
            probs = np.array(list(model[tuple(text[-2:])].values()))
            idx = np.argmax(probs)
            words = list(model[tuple(text[-2:])].keys()) #[idx]
            text.append(words[idx])
            print(words[idx])
    song = ' '.join([t for t in text if t])

    return song

def write_midifile(song):
    #time in ticks
    velocity = 90 #Same for JSB songs
    total_tick_time = 0
    #No channel info
    is_active = np.zeros(128, dtype=bool)


    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)


    for token in song.split(' '):
        if token == "<START>":
            pass

        if token == "SILENCE":
            #set every note to inactive
            is_active = np.zeros(128, dtype=bool)
            #todo

        if token.split('<')[0] == "TICKSHIFT":
            tick = re.search("<(.*?)>", token).group(1)
            tick = int(tick)
            total_tick_time = total_tick_time + tick

        if token.split('<')[0] == "NOTE_ON":
            note = re.search("<(.*?)>", token).group(1)
            is_active[int(note)] = True
            # print('note_on', note=int(note), velocity=velocity, time=0)
            print('note_on', 'note=',int(note), 'velocity=', velocity, 'time=',0)

        if token.split('<')[0] == "NOTE_OFF":
            note = re.search("<(.*?)>", token).group(1)
            is_active[int(note)] = False
            print('note_off', 'note=', int(note), 'velocity=',0, 'time=',0)

        if token == "<END>":
            pass
            #todo

    # track.append()

    # mid.save('sample_song_'+datetime.now().strftime("%H_%M_%S")+'.mid')
    return 0

if __name__ == "__main__":
    np.random.seed(10)
    # torch.manual_seed(0)
    datapath = r"C:\Users\isabelle\Documents\PythonScripts\ift6010-h21-team1\corpus.txt"

    corpus = get_sentences(datapath)
    model = get_ngram_model(corpus)
    song = generate_song(model)

    for token in song.split(' '):
        print(token)

    write_midifile(song)

