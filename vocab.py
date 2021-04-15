"""
Prepares vocabulary for generation
"""
import pickle
import os
from typing import List, Set


def get_vocab(path: str) -> List[str]:
    """Gathers the vocabulary for located at the path, will generate a vocab
    if it does not exist. The name of the vocab determines how it is generated.

    Args:
        path (str): Path to the vocab

    Returns:
        List[str]: List of tokens in the vocabulary
    """
    # create a new vocab, using pickle
    if not os.path.exists(path):
        folder = os.path.dirname(path)
        train_data_point = ""

        if "_musicautobot" in path:
            train_data_point = "train/text/70.mid"
        elif "_music21_default" in path:
            train_data_point = "train/music21 default/70.mid"
        else:
            raise Exception("Unknown encoding for music MIDIs")
        
        voc = create_vocab(os.path.join(folder, train_data_point))

        with open(path, mode='wb') as new_voc:
            pickle.dump(voc, new_voc)

        return voc

    # load vocab
    else:
        with open(path, mode='rb') as pickled_data:
            return pickle.load(pickled_data)


def create_vocab(path: str) -> List[str]:
    """Creates a vocab based on the name of the encoding folder inside
    each train, test, valid set. For example, 'text' or 'music21 default'.

    Args:
        path (str): Location of one of the train data point

    Returns:
        List[str]: [description]
    """
    voc : Set[str] = set()

    # Assume whitespace tokenization
    folder = os.path.dirname(path)

    for data_point in os.listdir(folder):
        with open(os.path.join(folder, data_point), mode='r') as data_file:
            for line in data_file:
                tokens = line.split()
                for token in tokens:
                    if token in voc:
                        pass
                    else:
                        voc.add(token)

    return list(voc)
