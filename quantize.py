#Quantize data into bits

import os
import pathlib
from mido import MidiFile, tick2second, second2tick
import numpy as np
import matplotlib.pyplot as plt
from _datetime import datetime

def get_maximum_song_length(datapath):
    lengths = []

    samples = os.listdir(datapath)
    time_list = [] #in ticks
    ticks_per_beats_list = []
    start = datetime.now()
    note = {}

    for sample in samples:
        mid = MidiFile(datapath + '/' + sample)
        lengths.append(mid.length)

        for i, track in enumerate(mid.tracks):
            TOTAL_TIME = 0
            TOTAL_TICK = 0  # 120 bpm, 100 ticks per beat
            notes = {}

            # Notes: 0 to 127
            for msg in track:

                if msg.type == 'set_tempo':
                    print("set_tempo ON")

                if msg.type == 'note_on' or msg.type == 'note_off':
                    # Access all information
                    dict = msg.dict()
                    channel, note, velocity, time = dict['channel'], dict['note'], dict['velocity'], dict['time']

                    time_list.append(time)
                    ticks_per_beats_list.append(mid.ticks_per_beat)
                    if str(note) not in notes.keys():
                        notes[str(note)] = True

    exctn_time =( datetime.now() - start).total_seconds()
    print("Took: {}".format(exctn_time))
    time_list = np.array(time_list)
    ticks_per_beats_list = np.array(ticks_per_beats_list)
    print("ticks per beats: {}".format(np.unique(ticks_per_beats_list)))
    print("time (ticks): {}".format(np.unique(time_list)))
    lengths = np.array(lengths)


    max, min, mean = lengths.max(), lengths.min(), lengths.mean()
    max_ticks = second2tick(second=max,ticks_per_beat=100, tempo=500000)
    min_ticks = second2tick(second=min,ticks_per_beat=100, tempo=500000)
    mean_ticks = second2tick(second=mean,ticks_per_beat=100, tempo=500000)
    print("Maximum length = {} ticks, minimum length = {} ticks, average length in ticks ={}".format(max_ticks, min_ticks, mean_ticks))
    print("Maximum length = {}s, minimum length = {}s, average length ={}s".format(max, min, mean))
    print("Unique notes:\n")
    for note in notes.keys():
        print(note)

    return max, min

class Quantizer:
    def __init__(self, datapath, timestep=1):
        self.datapath = datapath
        self.timestep = timestep

    def draw_piano_roll(self, sample, notes_dict):

        for k, v in notes_dict.items():
            x = [v[i:i + 2] for i in range(0, len(v), 2)]
            y = [int(k), int(k)]
            for pair in x:
                plt.plot([pair[0], pair[1]], [y, y], c='r')

        plt.title(sample)
        # plt.xlim(0,480)
        plt.show()

    def encode_midi(self, songpath):

        mid = MidiFile(songpath)
        TEMPO = 500000
        ticks_per_beat = mid.ticks_per_beat

        encoded_song = []
        notes ={}

        for i, track in enumerate(mid.tracks):
            encoded_song.append("<START>")
            TOTAL_TICK = 0  # 120 bpm, 100 ticks per beat
            temp_tick = 0
            is_active = np.zeros(128, dtype=bool)

            # Notes: 0 to 127
            for msg in track:

                if msg.type == 'note_on' or msg.type == 'note_off':
                    dict = msg.dict()
                    channel, note, velocity, time = dict['channel'], dict['note'], dict['velocity'], dict['time']

                    TOTAL_TICK = TOTAL_TICK + time

                    if str(note) not in notes.keys():
                        notes[str(note)] = [TOTAL_TICK]
                    else:
                        notes[str(note)].append(TOTAL_TICK)

                if msg.type == 'note_on':
                    # Case: Silence in between a pevious NOTE_OFF and current NOTE_ON event
                    if is_active.sum() == 0 and TOTAL_TICK-temp_tick !=0:
                        encoded_song.append("SILENCE")
                        encoded_song.append("TICKSHIFT<{}>".format(TOTAL_TICK-temp_tick))
                        temp_tick = TOTAL_TICK

                    encoded_song.append("NOTE_ON<{}>".format(note))
                    is_active[note] = True

                if msg.type == 'note_off':
                    if TOTAL_TICK-temp_tick ==0:
                        pass
                    else:
                        encoded_song.append("TICKSHIFT<{}>".format(TOTAL_TICK-temp_tick))
                        temp_tick = TOTAL_TICK
                    encoded_song.append("NOTE_OFF<{}>".format(note))
                    is_active[note] = False

        encoded_song.append("<END>")

        return encoded_song

    def build_corpus_file(self, directory_path, output_path, output_name):
        songs = os.listdir(directory_path)
        list_encoded_songs = []
        for song in songs:
            list_encoded_songs.append(self.encode_midi(directory_path+'/'+song))

        with open("corpus.txt", "a") as f:
            for encoded_song in list_encoded_songs:
                for word in encoded_song:
                    f.write("{} ".format(word))
                f.write("\n")

    def exploration1(self, datapath='/data/JSB Chorales/train', number_of_songs=5):

        # JSB chorales quantize into 120 ticks chunks

        entries = os.listdir(datapath)
        samples = np.random.choice(entries, size=number_of_songs)

        for sample in samples:
            mid = MidiFile(datapath+'/'+sample)
            TEMPO = 500000
            ticks_per_beat = mid.ticks_per_beat

            for i, track in enumerate(mid.tracks):
                TOTAL_TIME = 0
                TOTAL_TICK = 0 #120 bpm, 100 ticks per beat
                msg_counter = 0
                notes = {}

                #Notes: 0 to 127
                for msg in track:
                    msg_counter = msg_counter + 1
                    print(msg)

                    if msg.type == 'note_on' or msg.type == 'note_off':


                        # Access all information
                        dict = msg.dict()
                        channel, note, velocity, time = dict['channel'], dict['note'], dict['velocity'], dict['time']

                        # Convert time into seconds
                        timeSecs = tick2second(msg.time, mid.ticks_per_beat, TEMPO)
                        TOTAL_TIME = TOTAL_TIME + timeSecs
                        TOTAL_TICK = TOTAL_TICK + time

                        if str(note) not in notes.keys():
                            notes[str(note)] = [TOTAL_TIME]
                        else:
                            notes[str(note)].append(TOTAL_TIME)

                # Draw piano roll
                self.draw_piano_roll(sample=sample, notes_dict=notes)

            assert np.abs(TOTAL_TIME - mid.length) < 1.0, "Length of track not consitnt with {}".format(mid.length)
            print("Total time: {}".format(TOTAL_TIME))
            print("Number of timesteps: {}".format(TOTAL_TIME/(self.timestep)))

if __name__ == "__main__":
    np.random.seed(0)
    # torch.manual_seed(0)

    datapath = r"C:\Users\isabelle\Documents\PythonScripts\ift6010-h21-team1\data\JSB Chorales\test"
    quantizer = Quantizer(datapath=datapath)
    # quantizer.exploration1(datapath=datapath)
    # enc_midi = quantizer.encode_midi(songpath=r"C:\Users\isabelle\Documents\PythonScripts\ift6010-h21-team1\data\JSB Chorales\train\23.mid")
    # print(enc_midi)
    # quantizer.build_corpus_file(datapath, '', '')
    # default timestep = 1
    max, min = get_maximum_song_length(datapath)
    # print("Maximum length = {}s, minimum length = {}s".format(max, min))



    # quantizer.build_corpus(datapath=datapath, number_of_songs=5)



