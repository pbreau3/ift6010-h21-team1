from typing import List
from music21 import converter

from music21.stream import Score

def readMIDI(path: str) -> Score:
    """Read a midi file into a music21.Score object

    Args:
        path (str): Path to the midi file

    Returns:
        Score: music21.Score object
    """
    return converter.parse(path)

def scoreToText(score: Score) -> List[str]:
    """Converts a score into a list of string tokens

    Args:
        score (Score): music21.Score object

    Returns:
        List[str]: List of tokens
    """
    tokens: List[str] = []

def main():
    s = readMIDI("data/JSB Chorales/test/102.mid")

    
if __name__ == "__main__":
    main()