"""
Generate music from the ngram model
"""
from typing import List
from vocab import get_vocab
from generate import Generator
from interpython import call_python_version


class KenLMGenerator(Generator):
    """
    This wraps on top of the Generator class. It will use execnet to
    cross the bridge to Python 2
    """

    def __init__(self, path: str, vocab_path: str) -> None:
        self.model = call_python_version(
            "2.7", "kenlm_cpp.py", "get_model", [path])

        def probability_function(tokens: List[str]) -> float:
            return call_python_version("2.7", "kenlm_cpp.py", "get_log_probability",
                                       [self.model, " ".join(tokens)])

        super().__init__(self.model, probability_function=probability_function,
                                   vocab=get_vocab(vocab_path))


def get_kenlm_generator(model_path: str, vocab_path: str):
    return KenLMGenerator(model_path, vocab_path)


def main():
    pass


if __name__ == "__main__":
    main()
