u"""
Run as python 2
"""
from __future__ import with_statement
from __future__ import absolute_import
from numpy import average
import kenlm

# Loop on the entire test set for an average perplexity
import os
from io import open


def perplexity(path):
    model = kenlm.Model(path)
    test_set_folder = u"data/JSB Chorales/valid/text/"

    perplexities = []

    for test_file in os.listdir(test_set_folder):
        with open(os.path.join(test_set_folder, test_file)) as read_in:
            tokens = read_in.read()
            perplexities.append(model.perplexity(tokens))
    return perplexities

def show_ngram_perplexities():
    print u"number of test \t max \t min \t mean"
    for model in os.listdir(u"model/kenlm/"):
        if model.endswith(u"arpa"):
            continue
        print model
        perplexities = perplexity(os.path.join(u"model/kenlm/", model))
        print u'\t'.join([unicode(len(perplexities)), unicode(max(perplexities)), unicode(min(
            perplexities)), unicode(average(perplexities))])

# look at the perplexities
def main():
    show_ngram_perplexities()


if __name__ == u"__main__":
    main()