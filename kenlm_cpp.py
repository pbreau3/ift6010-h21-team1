u"""
Run as Python 2
"""
from __future__ import absolute_import
import kenlm

def get_model(path):
    u"""Get the KenLM model from disk

    Args:
        path (str): Path to the .bin kenlm model

    Returns:
        Any: A KenLM model class in Python 2
    """
    return kenlm.Model(path)

def get_perplexity(model, tokens):
    u"""Compute the perplexity of the model given the tokens

    Args:
        model (Any): A KenLM model class in Python 2
        tokens (str): One string containing all tokens to be computed
    """
    return model.perplexity(tokens)

def get_log_probability(model, sentence, bos = True, eos = False):
    u"""Get the log 10 probability of the sentence

    Args:
        model (Any): A KenLM class in Python 2
        sentence (str): One long string of tokens
        bos (bool, optional): Is the string including the start of the music? Defaults to True.
        eos (bool, optional): Is the string including the end of the music? Defaults to True.
    """
    return model.score(sentence, bos, eos)