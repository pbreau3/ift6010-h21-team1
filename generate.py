"""
Generation script

Serves as a template to implement functions
"""

from typing import Any, Callable, List


class Generator():
    
    def __init__(self, model: Any, probability_function: Callable[[List[str]], float],
    vocab: List[str]) -> None:
        """Creates a generator wrapper class

        Args:
            model (Any): Your language model
            probability_function (List[str] -> float): Must be a function that takes a list of 
                previous tokens and returns the probability of such list according to the model
            vocab (List[str]): The vocabulary of the language model
        """
        self.model = model
        self.probability_function = probability_function
        self.tokens : List[str] = []
        self.vocab = vocab

    def generate_all(self, token_limit = 200) -> List[str]:
        """Generate all tokens until limit

        Args:
            token_limit (int, optional): Limit of how many tokens. Defaults to 200.

        Returns:
            List[str]: Newly generated tokens
        """
        new_tokens : List[str] = []
        for _ in range(token_limit):
            new_token = self.next_most_likely_token()
            if new_token == "":
                break
            new_tokens.append(new_token)
            self.tokens.append(new_token)
        return new_tokens

    def generate_next(self) -> str:
        """Get the next generated token

        Returns:
            str: Next token
        """
        new_token : str = self.next_most_likely_token()
        self.tokens.append(new_token)
        return new_token

    def next_most_likely_token(self) -> str:
        """Compute the next most likely token, does not store it
        in the state list self.tokens

        Returns:
            str: Next most likely token. Empty string if nothing is likely.
        """
        highest_probability = 0
        most_likely_token = ""
        for token in self.vocab:
            probability = self.probability_function(self.tokens + [token])
            if probability > highest_probability:
                highest_probability = probability
                most_likely_token = token
        return most_likely_token