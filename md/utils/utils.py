import MDAnalysis as mda 
import MDAnalysis.transformations as trans
import os
from multiprocessing import Pool
from openbabel import openbabel
import tempfile


#trajectory utils
def remove_hydrogens(input_pdb, output_pdb):
    """
    Removes all hydrogens in a PDB
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdb")
    obErrorLog = openbabel.OBError
    openbabel.cvar.obErrorLog.SetOutputLevel(0)  # Suppress Open Babel output
    
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_pdb)
    
    mol.DeleteHydrogens()  # Remove all hydrogens
    
    obConversion.WriteFile(mol, output_pdb)

def write_trajectory(mda_universe, filename, sparsity = 1, start_frame=0, end_frame=-1):
    """
    Writes the mda.universe trajectory to a file.
    
    Parameters:
        mda_universe (MDAnalysis.Universe): The Universe containing the centered trajectory.
        filename (str): The output filename (e.g., "centered_trajectory.dcd").
        sparsity (int): Every nth frame will be written to traj
        start and end frame (int): self explanatory, default is whole trajectory 
    """
    with mda.Writer(filename, mda_universe.atoms.n_atoms) as writer:
        for ts in mda_universe.trajectory[start_frame:end_frame:sparsity]:
            writer.write(mda_universe.atoms)


def center_align_process_block(pdb_filename, traj_filename, start, stop, temp_filename):
    """
    Process a block of trajectory frames:
      - Load the Universe.
      - Process frames [start, stop) by centering the protein.
      - Write the centered frames to a temporary file.
    Parameters:
        pdb_filename (str): Path to the topology file (e.g. PDB).
        traj_filename (str): Path to the trajectory file.
        start (int): Starting frame index (inclusive).
        stop (int): Ending frame index (exclusive).
        temp_filename (str): Filename for the temporary output.
    Returns:
        str: The temporary filename written.
    """
    # Load the Universe for this block
    u = mda.Universe(pdb_filename, traj_filename)

    custom_vdw_radii = {"Na":1}    
    guesser = mda.guesser.default_guesser.DefaultGuesser(u, vdwradii=custom_vdw_radii)
    bonds = guesser.guess_bonds()
    u.add_TopologyAttr('bonds', bonds)

    ref = u.copy()
    
    ref_protein = ref.select_atoms('protein')
    ref_backbone = ref.select_atoms('backbone')
    ref_not_protein = ref.select_atoms('not protein')

    #transform frame 0
    ref_transformations = [trans.unwrap(ref_protein), trans.center_in_box(ref_protein), trans.wrap(ref_not_protein, compound="residues")]
    ref.trajectory.add_transformations(*ref_transformations)

    ref.trajectory[0] #set ref to frame 0 and run transformation
    
    protein = u.select_atoms('protein')
    backbone = u.select_atoms('backbone')
    not_protein = u.select_atoms('not protein')

    # Define transformations for u with the alignment step
    transformations = [trans.unwrap(protein), trans.center_in_box(protein), trans.fit_rot_trans(backbone, ref_backbone, weights = backbone.masses), trans.wrap(not_protein, compound="residues")]
    u.trajectory.add_transformations(*transformations)
    
    # Create a Writer for the temporary trajectory file.
    with mda.Writer(temp_filename, u.atoms.n_atoms) as writer:
        for frame in range(start, stop):
            u.trajectory[frame]  # read frame (transformations are applied here)
            writer.write(u.atoms)
    
    return temp_filename

def center_process_block(pdb_filename, traj_filename, start, stop, temp_filename):
    """
    Process a block of trajectory frames:
      - Load the Universe.
      - Process frames [start, stop) by centering the protein.
      - Write the centered frames to a temporary file.
    Parameters:
        pdb_filename (str): Path to the topology file (e.g. PDB).
        traj_filename (str): Path to the trajectory file.
        start (int): Starting frame index (inclusive).
        stop (int): Ending frame index (exclusive).
        temp_filename (str): Filename for the temporary output.
    Returns:
        str: The temporary filename written.
    """
    # Load the Universe for this block
    u = mda.Universe(pdb_filename, traj_filename)
    
    protein = u.select_atoms('protein')
    backbone = u.select_atoms('backbone')
    not_protein = u.select_atoms('not protein')

    # Define transformations for u with the alignment step
    transformations = [trans.unwrap(protein), trans.center_in_box(protein), trans.wrap(not_protein, compound="residues")]
    u.trajectory.add_transformations(*transformations)
    
    # Create a Writer for the temporary trajectory file.
    with mda.Writer(temp_filename, u.atoms.n_atoms) as writer:
        for frame in range(start, stop):
            u.trajectory[frame]  # read frame (transformations are applied here)
            writer.write(u.atoms)
    
    return temp_filename

def parallel_center_trajectory(pdb_filename, traj_filename, align, n_jobs=4, output_filename="centered_trajectory.dcd"):
    """
    Splits the trajectory into blocks and processes each block in parallel.
    After processing, the temporary files are concatenated into one final trajectory.
    Parameters:
        pdb_filename (str): Path to the topology file.
        traj_filename (str): Path to the trajectory file.
        n_jobs (int): Number of parallel blocks (jobs) to use.
        output_filename (str): Name of the final centered trajectory file.
    Returns:
        mda universe from the centered trajectory
    """
    # Load Universe once to get number of frames.
    u = mda.Universe(pdb_filename, traj_filename)
    n_frames = u.trajectory.n_frames
    frames_per_block = n_frames // n_jobs
    
    # Create tasks for each block and list to hold temp filenames.
    tasks = []
    temp_files = []

    for i in range(n_jobs):
        start = i * frames_per_block
        stop = (i + 1) * frames_per_block if i < n_jobs - 1 else n_frames # Last block takes all remaining frames.
        
        temp_filename = f"temp_centered_block_{i}.dcd"
        temp_files.append(temp_filename)
        tasks.append((pdb_filename, traj_filename, start, stop, temp_filename))

    # Use multiprocessing Pool to process tasks in parallel
    with Pool(n_jobs) as pool:
        if align == False:
            pool.starmap(center_process_block, tasks)
        elif align == True:
            pool.starmap(center_align_process_block, tasks)
    
    # Combine temporary files into the final output trajectory.
    with mda.Writer(output_filename, u.atoms.n_atoms) as writer:
        for temp_file in temp_files:
            temp_u = mda.Universe(pdb_filename, temp_file)
            for ts in temp_u.trajectory:
                writer.write(temp_u.atoms)
    
    #Clean up the temporary files.
    for temp_file in temp_files:
        os.remove(temp_file)
    
    #print(f"Centered trajectory written to {output_filename}")

    mda_universe = mda.Universe(pdb_filename, output_filename)
    return mda_universe






    