
# https://stackoverflow.com/questions/2460177/edit-distance-in-python#32558749
from __future__ import with_statement
from __future__ import absolute_import
from typing import List
from itertools import izip
from io import open


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = xrange(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_token_type(token):
    if u'n' in token:
        return u'n'
    elif u'd' in token:
        return u'd'
    elif token == u"xxsep":
        return u"xxsep"
    else:
        raise Exception(u"Unknown token type: {token}".format(token=token))

def token_by_token(expected, generated):
    u"""Compute the distance metric between two tokens
    
    Same token type: look for the difference between the associated number
    Ex: n84 vs n98 => |84-94| = 10

    Different token type: 100

    Args:
        expected (str): Expected token
        generated (str): Generated token

    Returns:
        int: Distance, 0 being the same
    """
    if expected == u"xxbos" or expected == u"xxpad":
        return 0

    expected_type = get_token_type(expected)
    generated_type = get_token_type(generated)

    if expected_type == u"xxsep" or generated_type == u"xxsep":
        return 100
    else:
        expected_int = int(expected[1:])
        generated_int = int(generated[1:])
        return abs(expected_int - generated_int)

def musicDistance(expected, generated):
    score = 0
    for expected_token, generated_token in izip(expected[2:], generated[2:]):
        score += token_by_token(expected_token, generated_token)
    return score

def fixed_premise_distance(model, voc, expected_file):
    with open(expected_file, u'r') as file:
        expected = file.read()
    expected = expected.split()

    from generate_kenlm_python2 import init_kenlm, set_kenlm_premise, next_most_likely_token

    init_kenlm(model, voc)

    score = 0

    premise = []
    for token in expected:
        set_kenlm_premise(premise)
        next_token = next_most_likely_token()

        score += token_by_token(token, next_token)

        premise.append(token)

    return score

def file_comparison():
    import sys
    if len(sys.argv) != 4:
        print u"Usage: distance.py expected_file generated_file [--levenshtein | --music]"
        exit(1)

    expected_file = sys.argv[1]
    generated_file = sys.argv[2]
    mode = sys.argv[3]

    with open(expected_file, u'r') as file:
        expected = file.read()
    with open(generated_file, u'r') as file:
        generated = file.read()
    
    # even the lengths of the files
    expected_tokens = expected.split()
    expected_size = len(expected_tokens)
    generated_tokens = generated.split()
    generated_size = len(generated_tokens)

    if expected_size < generated_size:
        generated_tokens = generated_tokens[:expected_size]
    else:
        expected_tokens = expected_tokens[:generated_size]

    if mode == u"--levenshtein":
        print levenshteinDistance(u" ".join(expected_tokens), u" ".join(generated_tokens))
    elif mode == u"--music":
        print musicDistance(expected_tokens, generated_tokens)

def main():
    import sys
    if len(sys.argv) != 4:
        print u"Usage: distance_python2.py model vocab test-token-file.txt"
        print u"Ex: distance_python2.py model/kenlm/order16.bin \"data/JSB Chorales/voc_musicautobot.voc\" test-token-file.txt"
        print u"Received: {argc} args".format(argc=len(sys.argv))
        exit(1)
    
    print fixed_premise_distance(sys.argv[1], sys.argv[2], sys.argv[3])

if __name__ == u"__main__":
    main()