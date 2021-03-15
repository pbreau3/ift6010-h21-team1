#Quantize data into bits

import os
import pathlib
from mido import MidiFile, tick2second
import numpy as np
import matplotlib.pyplot as plt

class Quantizer:
    def __init__(self, datapath, timestep=0):
        self.datapath = datapath
        self.timestep = timestep

    def build_corpus(self, datapath='/data/JSB Chorales/train', number_of_songs=5):

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
                # quanta_matrix = np.ones((128,1))
                # song_matrix = []
                x72 = []
                y = 60
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
                for k,v in notes.items():
                    x = [v[i:i + 2] for i in range(0, len(v), 2)]
                    y = [int(k), int(k)]
                    for pair in x:
                        plt.plot([pair[0], pair[1]], [y, y], c='r')
                plt.show()

            assert np.abs(TOTAL_TIME - mid.length) < 1.0, "Length of track not consitent with {}".format(mid.length)
            print("Total time: {}".format(TOTAL_TIME))

if __name__ == "__main__":
    datapath = r"C:\Users\isabelle\Documents\PythonScripts\ift6010-h21-team1\data\JSB Chorales\train"

    quantizer = Quantizer(datapath=datapath)
    quantizer.build_corpus(datapath=datapath, number_of_songs=5)


