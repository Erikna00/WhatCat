import MDAnalysis as mda 
import MDAnalysis.transformations as trans
import os
from multiprocessing import Pool
import json
from rdkit import Chem
from openbabel import openbabel
import tempfile
import warnings
#Filter biopython warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="Bio.Application")

###############
#Argparse utils
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


###############
#molecule utils
################

class FF_cache_reader():
    """
    Checks a molecule containing file (sdf, mol2) against a openmm forcefield cache JSON to see if it is there.
    Used to determine if we want to calculate charges via am1bcc with the gastreiger fallback option.
    Introduced due to non-converging SCF in am1bcc for GGPP.
    """
    def __init__(self, cache_file):
        self.cache = self.load_cache(cache_file)

    def load_cache(self, cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
        
    def get_canonical_smiles(self, molecule_file):
        return Chem.MolToSmiles(Chem.MolFromMolFile(molecule_file), canonical=True)
    
    def check_smiles_in_cache(self, molecule_smiles):
        "Check if a molecule's SMILES exists in the cache"
        for entry in self.cache:
            if 'smiles' in entry and entry['smiles'] == molecule_smiles:
                return True
        #if we fail all smiles checks we return false
        return False

    def check_molecule_in_cache(self, molecule_file):
        smiles = self.get_canonical_smiles(molecule_file)
        boolean = self.check_smiles_in_cache(smiles)
        return boolean


def prepare_ligand_md(file, pH = 7.4):
    """
    Uses Openbabel to convert a file to sdf
    Can optionally also reprotonate the molecule to match a certain total charge

    Parameters:
        pH : float 
            Indicates pH for protonation, None to indicate that file is correctlly protonated and charged
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
    """
    with mda.Writer(filename, mda_universe.atoms.n_atoms) as writer:
        for ts in mda_universe.trajectory[start_frame:end_frame:sparsity]:
            writer.write(mda_universe.atoms)


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
    ref_protein = ref.select_atoms('protein')
    ref_backbone = ref.select_atoms('backbone')
    ref_not_protein = ref.select_atoms('not protein')

    #transform frame 0
    ref_transformations = [trans.unwrap(ref_protein), trans.wrap(ref_not_protein, compound="residues"), trans.center_in_box(ref_protein), trans.wrap(ref_not_protein, compound="residues")]
    ref.trajectory.add_transformations(*ref_transformations)

    ref.trajectory[0] #set ref to frame 0 and run transformation
    
    protein = u.select_atoms('protein')
    backbone = u.select_atoms('backbone')
    not_protein = u.select_atoms('not protein')

    # Add centering transformations
    #We need to rewrap the PBC before fit_rot_trans to avoid ligand drift in binding site.
    transformations = [trans.unwrap(protein), trans.center_in_box(protein), trans.wrap(not_protein, compound="residues")]

    if align == True:
        #If aligning we add alignment transformations
        transformations += [trans.fit_rot_trans(backbone, ref_backbone, weights = backbone.masses), trans.wrap(not_protein, compound="residues")]

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

    