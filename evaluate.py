"""
Run as python 2
"""
from statistics import mean
import kenlm

# Loop on the entire test set for an average perplexity
import os


def perplexity(path: str):
    model = kenlm.Model(path)
    test_set_folder = "data/JSB Chorales/valid/text/"

    perplexities = []

    for test_file in os.listdir(test_set_folder):
        with open(os.path.join(test_set_folder, test_file)) as read_in:
            tokens = read_in.read()
            perplexities.append(model.perplexity(tokens))
    return perplexities

def show_ngram_perplexities():
    print("number of test \t max \t min \t mean")
    for model in os.listdir("model/kenlm/"):
        if model.endswith("arpa"):
            continue
        print(model)
        perplexities = perplexity(os.path.join("model/kenlm/", model))
        print(len(perplexities), max(perplexities), min(
            perplexities), mean(perplexities), sep='\t')

# look at the perplexities
def main():
    show_ngram_perplexities()


if __name__ == "__main__":
    main()