import numpy as np
import scipy.stats
import MDAnalysis as mda 
import pandas as pd
import os
import multiprocessing as mp
from MDAnalysis.analysis.distances import dist
import time

def compute_pairwise_distance_frame(atomgroup, pair_selections):
    """
    Compute pairwise distances for all specified atom pairs on the current frame.
    
    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        The atom group updated for the current frame.
    pair_selections : list of tuple(str, str)
        List of tuples, each containing two MDAnalysis selection strings.
    
    Returns
    -------
    np.ndarray
        1D array of distances for each specified atom pair for the current frame.
    """
    distances = []
    for sel1, sel2 in pair_selections:
        # Perform the selections on the current frame; note that since atomgroup is updated,
        # these selections reflect the current positions.
        a1 = atomgroup.select_atoms(sel1)
        a2 = atomgroup.select_atoms(sel2)
        if len(a1) != 1 or len(a2) != 1:
            raise ValueError(f"Each selection must match exactly one atom: {sel1}, {sel2} \n currentlly {a1.n_atoms} and {a2.n_atoms}")
        # Use MDAnalysis's 'dist' function; it returns an array, and the last element is the
        # distance between the two atoms.
        d = dist(a1, a2)[-1, 0]
        distances.append(d)
    return np.array(distances)

def compute_rmsf_chunk(pdb_filename, traj_filename, frame_indices, selection, ref_pdb):
    """
    Compute mean squared fluctuations for a subset of frames, applying on‐the‐fly alignment to ref_pdb.
    
    Parameters:
        pdb_filename (str): Path to topology file.
        traj_filename (str): Path to trajectory file.
        frame_indices (list of int): Frames assigned to this worker.
        selection (str): Atom selection string.
        ref_positions (np.ndarray): Reference positions (for the selected atoms) for alignment.
        
    Returns:
        np.ndarray: Sum of squared fluctuations per atom.
        int: Number of frames processed.
    """

    # Load Universe in each worker
    u = mda.Universe(pdb_filename, traj_filename)
    atoms = u.select_atoms(selection)

    # Load the reference structure from file and select atoms
    ref_u = mda.Universe(ref_pdb)
    ref_atoms = ref_u.select_atoms(selection)
    
    # Add transformation for on-the-fly alignment using the provided reference positions
    u.trajectory.add_transformations(mda.transformations.fit_rot_trans(atoms, ref_atoms))
    
    n_atoms = atoms.n_atoms
    sum_squared_flucts = np.zeros(n_atoms)
    
     # Process each frame in this chunk: the transformation is applied as frames are read.
    for frame in frame_indices:
        u.trajectory[frame]
        diff = atoms.positions - ref_atoms.positions
        sum_squared_flucts += np.sum(diff**2, axis=1)
    
    return sum_squared_flucts, len(frame_indices)


def compute_2d_rmsd_block(pdb_filename, traj_filename, selection, frame_pairs):
    """ 
    Compute a block of the upper-triangle RMSD matrix.
    
    Parameters:
        pdb_filename (str): Path to the topology file.
        traj_filename (str): Path to the trajectory file.
        selection (str): Atom selection string for RMSD calculation.
        frame_pairs (list of tuples): Pairs of frame indices (i, j) to compute RMSD.

    Returns:
        dict: Computed RMSD values, indexed by (i, j).
    """

    u = mda.Universe(pdb_filename, traj_filename)
    atoms = u.select_atoms(selection)
    
    rmsd_results = {}

    for i, j in frame_pairs:
        u.trajectory[i]
        ref_positions = atoms.positions.copy()

        u.trajectory[j]
        rmsd_results[(i, j)] = mda.analysis.rms.rmsd(atoms.positions, ref_positions)

    return rmsd_results

def radgyr(atomgroup, masses, total_mass=None):
    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates-center_of_mass)**2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)
    sq_x = np.sum(ri_sq[:,[1,2]], axis=1) # sum over y and z
    sq_y = np.sum(ri_sq[:,[0,2]], axis=1) # sum over x and z
    sq_z = np.sum(ri_sq[:,[0,1]], axis=1) # sum over x and y

    # make into array
    sq_rs = np.array([sq, sq_x, sq_y, sq_z])

    # weight positions
    rog_sq = np.sum(masses*sq_rs, axis=1)/total_mass

    # square root and return
    return np.sqrt(rog_sq)

