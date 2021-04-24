
# https://stackoverflow.com/questions/2460177/edit-distance-in-python#32558749
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_token_type(token: str) -> str:
    if 'n' in token:
        return 'n'
    elif 'd' in token:
        return 'd'
    elif token == "xxsep":
        return "xxsep"
    else:
        raise Exception("Unknown token type: {token}")

def token_by_token(expected: str, generated: str) -> int:
    """Compute the distance metric between two tokens
    
    Same token type: look for the difference between the associated number
    Ex: n84 vs n98 => |84-94| = 10

    Different token type: 100

    Args:
        expected (str): Expected token
        generated (str): Generated token

    Returns:
        int: Distance, 0 being the same
    """
    expected_type = get_token_type(expected)
    generated_type = get_token_type(generated)

    if expected_type != generated_type:
        return 100
    else:
        expected_int = int(expected[1:])
        generated_int = int(generated[1:])
        return abs(expected_int - generated_int)

def musicDistance(expected: str, generated: str) -> int:
    score = 0
    for expected_token, generated_token in zip(expected.split()[2:], generated.split()[2:]):
        score += token_by_token(expected_token, generated_token)
    return score

def main():
    import sys
    if len(sys.argv) != 4:
        print("Usage: distance.py expected_file generated_file [--levenshtein | --music]")
        exit(1)

    expected_file = sys.argv[1]
    generated_file = sys.argv[2]
    mode = sys.argv[3]

    with open(expected_file, 'r') as file:
        expected = file.read()
    with open(generated_file, 'r') as file:
        generated = file.read()
    
    if mode == "--levenshtein":
        print(levenshteinDistance(expected, generated))
    elif mode == "--music":
        print(musicDistance(expected, generated))

if __name__ == "__main__":
    main()