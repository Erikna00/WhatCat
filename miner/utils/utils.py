import numpy as np
import MDAnalysis as mda

def numpy_to_blankspace_sep_str(np_array):
    # Convert each element to a string and join them with a space
    np_string = ' '.join(map(str, np_array))
    return np_string

def load_universe(structure_file, trajectory_file=None):
    if trajectory_file:
        universe = mda.Universe(structure_file, trajectory_file)
    else:
        universe = mda.Universe(structure_file)
    return universe
