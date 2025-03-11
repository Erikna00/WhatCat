import numpy as np


def numpy_to_blankspace_sep_str(np_array):
    # Convert each element to a string and join them with a space
    np_string = ' '.join(map(str, np_array))
    return np_string
