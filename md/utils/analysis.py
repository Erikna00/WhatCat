import numpy as np
import scipy.stats
import MDAnalysis as mda 
import pandas as pd
import os
import multiprocessing as mp

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import dist
import time

def compute_pairwise_distance(atomgroup, pair_selections):
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

def calculate_pairwise_distances(pdb_file, traj_file, atom_pairs, **run_kwargs):
    """
    Calculate pairwise distances between specified atom pairs for a trajectory.
    Each element of `atom_pairs` should be a string with two selection
    queries separated by a comma.
    
    Parameters
    ----------
    pdb_file : str
        Path to the topology (PDB) file.
    traj_file : str
        Path to the trajectory file.
    atom_pairs : np.ndarray or list of str
        Array/list of strings specifying atom pairs in the format:
        "selection1, selection2"
    run_kwargs : dict
        Additional keyword arguments to pass to the analysis.run() method, for example:
        backend="multiprocessing", n_workers=4.
    
    Returns
    -------
    np.ndarray
        2D NumPy array of shape (num_frames, num_pairs) where each row contains the
        computed distances for the specified pairs in that frame.
    """
    # Convert each input string into a tuple of two selection strings.
    pair_selections = []
    for pair in atom_pairs:
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError("Each atom pair must be in the format: 'selection1, selection2'")
        pair_selections.append((parts[0], parts[1]))
    
    # Create the MDAnalysis Universe.
    u = mda.Universe(pdb_file, traj_file)
    
    # Set up AnalysisFromFunction.
    # Here, we pass u.trajectory as the trajectory to iterate over and u.atoms as the AtomGroup
    # to be updated on each frame. The 'pair_selections' tuple is passed as an argument to our function.
    analysis = mda.analysis.base.AnalysisFromFunction(compute_pairwise_distance, u.trajectory, u.atoms, pair_selections)
    analysis.run(**run_kwargs)
    
    # The per-frame distances are stored in analysis.results.timeseries.
    # This array will have shape (num_frames, num_pairs).
    return analysis.results.timeseries

def compute_rmsf_chunk(pdb_filename, traj_filename, frame_indices, selection, ref_pdb):
    """
    Compute mean squared fluctuations for a subset of frames, applying on‐the‐fly alignment.
    
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

def calc_rmsf_parallel(pdb_filename, traj_filename, n_jobs=4):
    """
    Compute RMSF with on-the-fly trajectory alignment using MDAnalysis transformations, 
    ensuring low memory usage by processing frames in parallel.
    
    Parameters:
        pdb_filename (str): Path to topology file.
        traj_filename (str): Path to trajectory file.
        n_jobs (int): Number of parallel workers.
    
    Returns:
        np.ndarray: Computed RMSF values per residue.
    """

    # Load Universe once to compute the average structure.
    u = mda.Universe(pdb_filename, traj_filename)
    selection = "protein and name CA"
    
    # Compute the average structure (for alignment reference)
    avg_struct = mda.analysis.align.AverageStructure(u, u, select=selection, ref_frame=0).run()

    # Set unit cell dimensions from the first frame before writing
    avg_struct.results.universe.dimensions = u.trajectory[0].dimensions

    # Write the average structure to file so workers can load it
    ref_pdb = f"{pdb_filename.replace(".pdb", "")}_avg_structure.pdb"
    avg_struct.results.universe.atoms.write(ref_pdb)
    while not os.path.exists(ref_pdb):
        time.sleep(0.1)
    
    n_frames = u.trajectory.n_frames
    
    # Split frame indices evenly among workers.
    frame_chunks = np.array_split(range(n_frames), n_jobs)
    
    with mp.Pool(n_jobs) as pool:
        results = pool.starmap(
            compute_rmsf_chunk,
            [(pdb_filename, traj_filename, list(chunk), selection, ref_pdb)
             for chunk in frame_chunks]
        )
    
    total_squared_flucts = None
    total_frames = 0
    # Aggregate results from all workers.
    for sum_sq, n in results:
        if total_squared_flucts is None:
            total_squared_flucts = sum_sq
        else:
            total_squared_flucts += sum_sq
        total_frames += n
    
    rmsf_values = np.sqrt(total_squared_flucts / total_frames)
    
    # Save RMSF as B-factors in a PDB file.

    u_out = mda.Universe(pdb_filename)
    u_out.add_TopologyAttr('tempfactors')
    protein = u_out.select_atoms("protein")
    
    # Assign computed RMSF values to residues (assuming one RMSF per CA atom)
    for residue, r_value in zip(protein.residues, rmsf_values):
        residue.atoms.tempfactors = r_value
    
    u_out.atoms.write(f"{pdb_filename}")
    
    return rmsf_values


def compute_rmsd_block(pdb_filename, traj_filename, selection, frame_pairs):
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

def parallel_2d_rmsd(pdb_filename, traj_filename, selection="protein", n_jobs=4):
    """ 
    Compute the full 2D RMSD matrix efficiently using process-based parallelism.
    
    Parameters:
        pdb_filename (str): Path to the topology file.
        traj_filename (str): Path to the trajectory file.
        selection (str): Atom selection string for RMSD calculation.
        n_jobs (int): Number of parallel jobs.

    Returns:
        np.ndarray: The computed symmetric RMSD matrix.
    """
    u = mda.Universe(pdb_filename, traj_filename)
    n_frames = u.trajectory.n_frames

    # Generate all (i, j) pairs for the upper triangle where j > i
    frame_pairs = [(i, j) for i in range(n_frames) for j in range(i + 1, n_frames)]
    split_pairs = np.array_split(frame_pairs, n_jobs)  # Distribute pairs across jobs

    with mp.Pool(n_jobs) as pool:
        #TODO revise all parallelization here and in RMSF to start more reliablly
        results_list = pool.starmap(compute_rmsd_block, [(pdb_filename, traj_filename, selection, list(pairs)) for pairs in split_pairs])

    # Assemble the full symmetric RMSD matrix
    rmsd_matrix = np.zeros((n_frames, n_frames))

    for results in results_list:
        for (i, j), value in results.items():
            rmsd_matrix[i, j] = value  # Upper triangle
            rmsd_matrix[j, i] = value  # Mirror to lower triangle

    return rmsd_matrix


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

def equillibration_check(data_df, reporting_time, dt = 100):
    """
    Uses a linear regression to find out if the simulation is properlly equillibrated
    """
    #Panda step 0 corresponds to the first step after equillibration
    analyzed_snaps = int((dt / reporting_time) +1) #how many reports are within 100 ps, +1 because noninclusive slicing
    step = data_df["Step"][0:analyzed_snaps]  # Produces a Pandas Series
    potential_energy = data_df["Potential Energy (kJ/mole)"][0:analyzed_snaps]
    temperature = data_df["Temperature (K)"][0:analyzed_snaps]
    volume = data_df["Box Volume (nm^3)"][0:analyzed_snaps]

    #calculate if system shows trends in some direction which indicates a too short equillibration
    regr_energy = scipy.stats.linregress(step, potential_energy)
    regr_volume = scipy.stats.linregress(step, volume)
    regr_temp = scipy.stats.linregress(step, temperature)
    print(f"R^2 for first 100 ps is good.\nenergy: {regr_energy.rvalue ** 2} \nvolume: {regr_volume.rvalue ** 2} \ntemp: {regr_temp.rvalue ** 2}")

