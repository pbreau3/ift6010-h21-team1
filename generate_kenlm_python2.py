u"""
Generate music from the ngram model
"""
from __future__ import absolute_import
import kenlm
from vocab_python2 import get_vocab

KenLM_instance = None

class Generator(object):

    def __init__(self, model, probability_function,
                 vocab):
        u"""Creates a generator wrapper class

        Args:
            model (Any): Your language model
            probability_function (List[str] -> float): Must be a function that takes a list of 
                previous tokens and returns the probability of such list according to the model
            vocab (List[str]): The vocabulary of the language model
        """
        self.model = model
        self.probability_function = probability_function
        self.tokens = []
        self.vocab = vocab

    def generate_all(self, token_limit=200):
        u"""Generate all tokens until limit

        Args:
            token_limit (int, optional): Limit of how many tokens. Defaults to 200.

        Returns:
            List[str]: Newly generated tokens
        """
        new_tokens = []
        for _ in xrange(token_limit):
            new_token = self.next_most_likely_token()
            if new_token == u"":
                break
            new_tokens.append(new_token)
            self.tokens.append(new_token)
        return new_tokens

    def generate_next(self):
        u"""Get the next generated token

        Returns:
            str: Next token
        """
        new_token = self.next_most_likely_token()
        self.tokens.append(new_token)
        return new_token

    def next_most_likely_token(self):
        u"""Compute the next most likely token, does not store it
        in the state list self.tokens

        Returns:
            str: Next most likely token. Empty string if nothing is likely.
        """
        highest_probability = highest_probability = float('-inf')
        most_likely_token = u""
        for token in self.vocab:
            probability = self.probability_function(self.tokens + [token])
            if probability > highest_probability:
                highest_probability = probability
                most_likely_token = token
        return most_likely_token
    
    def set_tokens(self, tokens):
        self.tokens = tokens


class KenLMGenerator(Generator):
    u"""
    This wraps on top of the Generator class. It will use execnet to
    cross the bridge to Python 2
    """

    def __init__(self, path, vocab_path):
        self.model = kenlm.Model(path)

        def probability_function(tokens):
            return self.model.score(u" ".join(tokens))

        super(KenLMGenerator, self).__init__(self.model, probability_function=probability_function,
                         vocab=get_vocab(vocab_path))


def get_kenlm_generator(model_path, vocab_path):
    return KenLMGenerator(model_path, vocab_path)

def init_kenlm(model_path, vocab_path):
    global KenLM_instance
    KenLM_instance = KenLMGenerator(model_path, vocab_path)

def set_kenlm_premise(tokens):
    global KenLM_instance
    KenLM_instance.set_tokens(tokens)

def generate_next():
    """It will add the generation to the token log for the premise

    Returns:
        [str]: Next token
    """
    global KenLM_instance
    return KenLM_instance.generate_next()

def next_most_likely_token():
    global KenLM_instance
    return KenLM_instance.next_most_likely_token()

def main():
    import sys
    model = sys.argv[1]
    vocab = sys.argv[2]
    premise = sys.argv[3]
    print model
    gen = get_kenlm_generator(model, vocab)
    gen.set_tokens(premise.split())
    print "Premise: " + premise
    print " ".join(gen.generate_all())


if __name__ == u"__main__":
    main()
