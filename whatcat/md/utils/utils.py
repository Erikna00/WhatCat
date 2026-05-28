import MDAnalysis as mda 
import MDAnalysis.transformations as trans
from openbabel import openbabel

import pandas as pd
import numpy as np

import os
import re
from multiprocessing import Pool
import json
import tempfile
import warnings

#Filter biopython warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="Bio.Application")

###############
#String handling utils
###############

def str_to_bool(string):
    "converts a string to bool"
    if string == "True" or string == "true":
        return True
    elif string == "False" or string == "false":
        return False
    else:
        raise ValueError("Bad input to a boolean field")

def prepend_list(list, prefix):
    """
    Adds a prefix to each element in a list of strings
    """

    for i in range(len(list)):
        list[i] = prefix + list[i]

    return list

def strip_str_from_list(list, remove_str):
    """
    strips the string from all elements in a list of strings if present
    """
    list2 = []
    for i in range(len(list)):
        list2.append(list[i].replace(remove_str, ""))

    return list2

def css_to_list(string):
    """
    Converts a comma separated string to a list of strings
    """

    list = [s.strip() for s in string.split(',') if s.strip()]

    return list

def list_to_css(lst):
    """
    Converts a list of stuff to a comma separated string
    """

    if len(lst) == 0:
        raise ValueError("List is empty, cannot convert to CSS")

    css_string = ", ".join([str(i) for i in lst])

    return css_string

def strip_mda_selection(selection_string, spaces_to_underscores = True):
    """
    Remove 'name ' or 'resname ' prefixes from an atom selection string.

    Parameters:
        selection_string : str or list of str
            The original atom selection string.
        spaces_to_underscores : bool
            If to replace spaces with underscores
    
    Returns:
        str or list depending on type of selection list: 
            The cleaned atom selection string.
    """
    undesirables = {"resname ":"", "name ":"", "and ":"", " ":"_" }

    if spaces_to_underscores == False:
        undesirables[" "] = " "

    if type(selection_string) == list:
        output = []
        for selection in selection_string:
            for word in undesirables.keys():
                selection = selection.replace(word, undesirables[word])
            output.append(selection)

    elif type(selection_string) == str:
        for word in undesirables.keys():
            selection_string = selection_string.replace(word, undesirables[word])
        output = selection_string
    
    else:
        raise ValueError("selection_string must be a string or list of strings")
    
    return output    

class Logfile_writer:
    """
    A simple filewriter class that writes strings to a specified log file.
    """
    def __init__(self, logfile, mode="a", remove_existing = True):
        """
        Initializes the Logger with a logfile path and mode.

        Parameters:
            logfile : str
                Path to the log file.
            mode : str
                File mode for writing (default is "a" for append).
            remove_existing : bool
                If True, removes the existing log file if it exists.
                Useful when rerunning the same analysis to avoid appending to old logs.
        """

        if os.path.exists(logfile) and remove_existing == True:
            os.remove(logfile)
        self.logfile = logfile
        self.mode = mode

    def write(self, string):
        """
        Writes the string to the log file.

        Parameters:
            string : str
                The string to write to the log file.
        """
        with open(self.logfile, self.mode) as f:
            f.write(string + "\n")


class Ligand_namer():
    """
    Class to generate PDB compatible names for ligands.
    Ensures names are three characters or shorter, using a counter for unknown ligands.
    """
    def __init__(self):
        self.unknown_ligands = 0

    def name_ligand(self, ligand_file):
        """
        Generates a three symbol (or shorter) name for a ligand provided from a file.
        
        Parameters:
            ligand_file : str 
                The path to the ligand file (e.g., "ligand.sdf").
        
        Returns:
            str : A PDB compatible name for the ligand.
        """

        #read name ensuring uppercase
        ligand_file = os.path.splitext(os.path.basename(ligand_file))[0].upper()
        
        # Automatically set as resname if it's exactly 3 letters or numbers long and no explicit resname is provided
        if len(re.findall(r'[A-Z0-9]', ligand_file)) <= 3:
            ligand_name = ligand_file
        else:
            ligand_name = (f"UN{self.unknown_ligands}")
            self.unknown_ligands += 1

        return ligand_name


##################
#Metadynamics utils
##################

def atom_idx_from_selection(selection_string, topology):
    """
    Converts a selection string to a list of atom indices.
    
    Parameters:
        selection_string : str 
            A string containing the selection (e.g., "name CA and resid 1").
        topology : PDB or openmm.simulation.topology
            The topology of the system, used for building a MDAnalysis universe
    
    Returns:
        list : A list of atom indices corresponding to the selection.
    """

    u = mda.Universe(topology)
    selected_atoms = u.select_atoms(selection_string)

    if len(selected_atoms) != 1:
        raise ValueError(f"Selection '{selection_string}' did not return exactly one atom. Returned {len(selected_atoms)} atoms.")

    return selected_atoms.indices.tolist()[0]

def platform_device_sanitizer(platform_str):
    """ 
    Takes a platform selection string and corrects capitalization as well as splitting out the device number
    if given. Otherwise defaults to 0.

    Parameters:
        platform_str : str

    Returns:
        platform_sanitized : device name
        device_index_str : comma separated string for feeding to openmm
    """
    #Start by extracting device id
    platform_sanitized = platform_str.split("_")[0].upper()
    
    try:
        device_index_str = platform_str.split("_")[1]
    except:
        device_index_str = "0"

    if platform_sanitized == "OPENCL":
        platform_sanitized = "OpenCL"

    return platform_sanitized, device_index_str


def colvar_sanitizer(colvar_parameter):
    """
    Takes the colvar parameters list and ensures it is min, max, width ordered
    This allows user to input max, min, width or min, max, width

    Parameters:
        colvar_parameter : list 
            A list of three floats representing the colvar parameters.
    Returns:
        list : A sanitized list of colvar parameters in the order [min, max, width].
    """
    colvar_updated = []

    for colvar in colvar_parameter:
        if len(colvar) != 3:
            raise ValueError(f"Colvar parameter {colvar} does not have exactly three elements.")

        min_val = min(colvar[0], colvar[1])
        max_val = max(colvar[0], colvar[1])
        width = colvar[2]

        colvar_updated.append([min_val, max_val, width])

    return colvar_updated

def metadynamics_unit_finder(atom_indicies):
    """
    Finds the unit of a colvar based on length
    """
    if len(atom_indicies) == 2:
        return "Ångström"
    elif len(atom_indicies) == 3 or len(atom_indicies) == 4:
        return "degrees"
    else:
        raise ValueError(f"Cannot determine unit for {len(atom_indicies)} atoms. Only 2-4 atoms are supported.")
    
def metadynamics_pes_convergence_reader(pdb_name):
    """
    Reads assorted convergence files from a metadynamics run.
    
    Parameters:
        pdb_name : str 
            The base name of the PDB file used in the metadynamics run.
    """

    pes_rmsd_lst = pd.read_csv(f"{pdb_name}_mtd_convergence.csv")["PES RMSD (kJ/mol)"].to_numpy().tolist()
    simulation_dump_lst = pd.read_csv(f"{pdb_name}_mtd_convergence.csv")["Simulation time (ns)"].to_numpy().tolist()

    if os.path.exists(f"{pdb_name}_metadynamics_pes.csv"):
        # Try to read as 2D (with index and header)
        pes_df = pd.read_csv(f"{pdb_name}_metadynamics_pes.csv", header=0)
        if pes_df.shape[1] > 2:
            # 2D case: drop index column if present
            pes_last = pes_df.set_index(pes_df.columns[0]).to_numpy()
        elif pes_df.shape[1] == 2:
            # 1D case: just get the free_energy column
            pes_last = pes_df["free_energy"].to_numpy()
        else:
            raise ValueError("Unexpected CSV format for metadynamics PES.")

    elif os.path.exists(f"{pdb_name}_metadynamics_pes.npy"):
        pes_last = np.load(f"{pdb_name}_metadynamics_pes.npy")
    
    return pes_rmsd_lst, simulation_dump_lst, pes_last

###############
#molecule utils
################

class FF_cache_reader():
    """
    Checks a molecule containing file (sdf, mol2) against a openmm forcefield cache JSON to see if it is there.
    Used to determine if we want to calculate charges via am1bcc with the gastreiger fallback option.
    """
    def __init__(self, cache_file):
        self.cache = self.load_cache(cache_file)

    def load_cache(self, cache_file):
        """
        Returns a nested dictionary representation of the cache json
        """
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def check_smiles_in_cache(self, molecule_smiles, forcefield):
        """
        Check if a molecule's SMILES exists in the cache
        
        Parameters:
            molecule_smiles : str 
                The SMILES string of the molecule to check. Needs to be in openff format
            forcefield : str
                The forcefield block to check in the cache, corresponds to the small molecule force field used during parametrization.
        """

        ff_block = self.cache[forcefield]
        exists = any(molecule_smiles in mol_data["smiles"] for mol_data in ff_block.values())
        return exists


def prepare_ligand_md(file, pH = 7.4):
    """
    Uses Openbabel to convert a file to sdf
    Can optionally also reprotonate the molecule to match a certain total charge

    Parameters:
        pH : float 
            Indicates pH for protonation, None to indicate that file is correctlly charged
    """

    #split filename
    ext = os.path.splitext(file)[-1].lstrip(".")
    name = os.path.splitext(file)[0].lstrip(".")
    
    # Initialize Open Babel conversion
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats(ext, "sdf")

    # Create a molecule object
    mol = openbabel.OBMol()

    # Read the molecule from the SDF file
    if not obConversion.ReadFile(mol, file):
        raise ValueError(f"Failed to read molecule from {file}")
    
    if pH != None:
        mol.CorrectForPH(pH)
        
    mol.AddHydrogens()

    print(f"\nAdjusting charges for {name}")
    print(f"Total charge infeered as {mol.GetTotalCharge()}\n")

    # Write the molecule with assigned charges to the SDF file
    obConversion.WriteFile(mol, f"{name}.sdf")
    return f"{name}.sdf"

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

#################
#trajectory utils
#################

def write_trajectory(mda_universe, filename, sparsity = 1, start_frame=0, end_frame=-1):
    """
    Writes the mda.universe trajectory to a file.
    
    Parameters:
        mda_universe : MDAnalysis.Universe 
            The Universe containing the centered trajectory.
        filename : str 
            The output filename (e.g., "centered_trajectory.dcd").
        sparsity : int 
            Every nth frame will be written to traj
        start and end frame : int 
            self explanatory, default is whole trajectory 

    Returns:
        list : list of time points corresponding to the frames in the sparse trajectory
    """
    sparse_time = []
    with mda.Writer(filename, mda_universe.atoms.n_atoms) as writer:
        for ts in mda_universe.trajectory[start_frame:end_frame:sparsity]:
            writer.write(mda_universe.atoms)
            sparse_time.append(ts.time)

    return sparse_time


def center_align_process_block(structure_filename, traj_filename, start, stop, temp_filename, align):
    """
    Process a block of trajectory frames:
      - Load the Universe.
      - Process frames [start, stop) by centering (and aligning) the protein.
      - Write the centered frames to a temporary file.

    Parameters:
        structure_filename : str 
            Path to the topology file (e.g. PDB).
        traj_filename :str 
            Path to the trajectory file.
        start : int
            Starting frame index (inclusive).
        stop : int 
            Ending frame index (exclusive).
        temp_filename :str
            Filename for the temporary output.
        align :bool 
            Wether to align protein to itself over traj or not. Necessary for some RMSD and RMSF dependant analysis

    Returns:
        str: The temporary filename written.
    """
    # Load the Universe for this block
    u = mda.Universe(structure_filename, traj_filename)

    ref = u.copy()
    ref_protein = ref.select_atoms("protein or resname CA MG ZN FE CU MN")
    ref_chain_a = ref.select_atoms("protein and chainid 1 A")
    ref_backbone = ref.select_atoms("backbone")
    ref_sel_all = ref.select_atoms("all")

    #transform frame 0 to have it availible for all processes for alignment
    ref_transformations = [trans.unwrap(ref_protein), trans.center_in_box(ref_chain_a), trans.wrap(ref_protein, compound="fragments"), 
                           trans.center_in_box(ref_protein), trans.wrap(ref_sel_all, compound="fragments")]
    ref.trajectory.add_transformations(*ref_transformations)

    ref.trajectory[0] #set ref to frame 0 and run transformation
    
    protein = u.select_atoms("protein or resname CA MG ZN FE CU MN")
    chain_a = u.select_atoms("protein and chainid 1 A")
    backbone = u.select_atoms("backbone")
    sel_all = u.select_atoms("all")

    # Add centering transformations
    #We need to rewrap the PBC before alignment to avoid ligand drift in binding site.

    #First we unwrap the PDC 
    #Then we center chain A to avoid issues with multidomain proteins ending up on three corners of the PBC
    #thus not centering the protein as a whole
    #Then we wrap the protein to get all protein chains inside the box
    #After which we center the whole protein and wrap everything into box
    #Not complicated at all ;)

    transformations = [trans.unwrap(protein), trans.center_in_box(chain_a), trans.wrap(protein, compound = "fragments"), 
                       trans.center_in_box(protein), trans.wrap(sel_all, compound="fragments")]

    if align == True:
        #Append alignment as well as subsequent wrapping to get a neat box
        transformations += [trans.fit_rot_trans(backbone, ref_backbone, weights = backbone.masses), trans.wrap(sel_all, compound="fragments")]

    u.trajectory.add_transformations(*transformations)
    
    # Create a Writer for the temporary trajectory file.
    with mda.Writer(temp_filename, u.atoms.n_atoms) as writer:
        for frame in range(start, stop):
            u.trajectory[frame]  # read frame (transformations are applied here)
            writer.write(u.atoms)
    
    return temp_filename

def parallel_center_trajectory(structure_filename, traj_filename, align, n_jobs=4, output_filename="centered_trajectory.dcd"):
    """
    Splits the trajectory into blocks and processes each block in parallel.
    After processing, the temporary files are concatenated into one final trajectory.

    Parameters:
        structure_filename (str): Path to the topology file. Also accepts OPENMM topologies which also includes bonding information.
        traj_filename (str): Path to the trajectory file.
        n_jobs (int): Number of parallel blocks (jobs) to use.
        output_filename (str): Name of the final centered trajectory file.

    Returns:
        mda universe from the centered trajectory
    """
    # Create a temporary directory inside the current working directory
    script_dir = os.getcwd()  # Get the current directory where the script is called
    with tempfile.TemporaryDirectory(dir=script_dir) as tmpdir:
        # Load Universe once to get number of frames.
        u = mda.Universe(structure_filename, traj_filename)
        n_frames = u.trajectory.n_frames
        frames_per_block = n_frames // n_jobs
        
        # Create tasks for each block and list to hold temp filenames.
        tasks = []
        temp_files = []

        # Adjust n_jobs to avoid creating more jobs than frames
        n_jobs = min(n_jobs, n_frames)
        
        # Create tasks for each block and list to hold temp filenames.
        tasks = []
        temp_files = []

        # Standard block processing
        if n_frames >= n_jobs:  
            frames_per_block = n_frames // n_jobs
            
            for i in range(n_jobs):
                start = i * frames_per_block
                stop = (i + 1) * frames_per_block if i < n_jobs - 1 else n_frames  # Last block takes all remaining frames.
                
                # Generate the temporary filename inside the tmpdir
                temp_filename = os.path.join(tmpdir, f"temp_centered_block_{i}.dcd")
                temp_files.append(temp_filename)
                tasks.append((structure_filename, traj_filename, start, stop, temp_filename, align))
        
        # If frames < n_jobs, assign one frame per job
        else:  
            for i in range(n_frames):
                start = i
                stop = i + 1
                
                temp_filename = os.path.join(tmpdir, f"temp_centered_block_{i}.dcd")
                temp_files.append(temp_filename)
                tasks.append((structure_filename, traj_filename, start, stop, temp_filename, align))

        # Use multiprocessing Pool to process tasks in parallel
        with Pool(n_jobs) as pool:
            pool.starmap(center_align_process_block, tasks)
        
        # Combine temporary files into the final output trajectory.
        u = mda.Universe(structure_filename, temp_files)
        u.atoms.write(output_filename, frames="all")
        
        # Cleanup is handled by TemporaryDirectory (all files in tmpdir are removed automatically)

    # Return the Universe of the final output trajectory
    mda_universe = mda.Universe(structure_filename, output_filename)
    return mda_universe

def average_structure_process_block(structure_filename, traj_filename, selection, start, stop, temp_filename):
    """
    Computes the average structure for a block of trajectory frames.
    Part of the parallel average structure calculation.
    Utility function for parallel_average_structure.

    Parameters:
        structure_filename : str 
            Path to the topology file (e.g. PDB).
        traj_filename :str 
            Path to the trajectory file.
        selection : str 
            Atom selection string for averaging.
        start : int
            Starting frame index (inclusive).
        stop : int 
            Ending frame index (exclusive).

    Returns:
        str: The temporary pdb file with the block average structu.
    """

    #Write the range of frames to be analyzed to a temporary file
    temp_traj = f"{temp_filename}.dcd"

    write_trajectory(mda.Universe(structure_filename, traj_filename), temp_traj, start_frame=start, end_frame=stop)

    # Load the Universe for this block
    u = mda.Universe(structure_filename, temp_traj)

    result = mda.analysis.align.AverageStructure(u, select=selection, ref_frame=0).run()
    avg_u = result.universe

    # Write the average structure to a temporary PDB file
    avg_pdb = f"{temp_filename}.pdb"
    avg_u.atoms.write(avg_pdb)

    return avg_pdb

def parallel_average_structure(structure_filename, traj_filename, selection, n_jobs, output_filename = None):
    """
    Computes the average structure of a trajectory in parallel by splitting it into blocks.
    Each block's average structure is computed and then combined to get the final average.

    Parameters:
        structure_filename (str): Path to the topology file.
        traj_filename (str): Path to the trajectory file.
        selection (str): Atom selection string for averaging.
        n_jobs (int): Number of parallel blocks (jobs) to use.
        output_filename (str): Name of the averaged structures PDB file. 
                               default is None which means no file printed.
    
    Returns:
        mda universe from the average structure
    """

    # Load Universe once to get number of frames.
    u = mda.Universe(structure_filename, traj_filename)
    n_frames = u.trajectory.n_frames

    # If we have enough frames to justify parallel processing
    if n_frames >= 10 * n_jobs:  
        # Create a temporary directory inside the current working directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create tasks for each block and list to hold temp filenames.
            tasks = []

            # Standard block processing
            frames_per_block = n_frames // n_jobs
            
            for i in range(n_jobs):
                start = i * frames_per_block
                stop = (i + 1) * frames_per_block if i < n_jobs - 1 else n_frames  # Last block takes all remaining frames.
                
                # Generate the temporary filename inside the tmpdir
                temp_filename = os.path.join(tmpdir, f"avg_block_{i}")
                tasks.append((structure_filename, traj_filename, selection, start, stop, temp_filename))

            # Use multiprocessing Pool to process tasks in parallel
            with Pool(n_jobs) as pool:
                results = pool.starmap(average_structure_process_block, tasks)
            
            u = mda.Universe(results[0], results)
            
            # Combine temporary files into the final averaging calculation which is run in serial
            result = mda.analysis.align.AverageStructure(u, select=selection, ref_frame=0, 
                                                        filename=output_filename).run()
            avg_u = result.universe

            # Cleanup is handled by TemporaryDirectory (all files in tmpdir are removed automatically)
    
    # If we don't have enough frames to justify parallel processing, do it in serial
    else:  
        result = mda.analysis.align.AverageStructure(u, select=selection, ref_frame=0, filename=output_filename).run()
        avg_u = result.universe

    return avg_u
