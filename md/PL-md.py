from openmm.app import *
from openmm import *
from openmm.unit import *
import sys
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer
import argparse
from utils import utils, analysis, plot
import re

#only needed for devwork
import importlib
importlib.reload(utils)

import warnings
# suppress some MDAnalysis warnings when writing PDB files as well as the DCD timestep warning
warnings.filterwarnings('ignore')

#Start the command line parser
parser = argparse.ArgumentParser(
                    prog='POS MD script',
                    description=(
                    "This script sets up a OPENMM  simulation of a protein (amber ff14)," 
                    "any small molecules/cofactors (Sage 2.2.1) as well as  water/ions using a 12-6 model (tip3pfb)."
                    "If using --pdbfix True then the structure will be prepared automatically. Otherwise prep is fully manual"),
                    epilog='Use with care and acknowledge Erik Sundén and the Per-Olof Syrén group at KTH Sweden')

parser.add_argument("pdb", type = str, help = "dockpreped PDB structure of the structure you want to simulate, including ligands") 
parser.add_argument("--pdbfix", type = str, default = "True", help = ("True or False depending on if your structure shall be PDBfixed." 
                                                                     "default = True \n good if you have a SEQRES and unresolved loops."
                                                                     "BEWARE, this function also removes all protein hydrogens and readds them" ))
parser.add_argument("-l", "--lig", type = str, action="append", default = None, help = ("optional parameter, SDF file containing all nonstandard ligands and cofactors." 
                                                                       "Convenientlly produced by drawing in chemdraw and exporting as SDF then docking with added hydrogens."
                                                                       "This is easilly done by checking ChimeraX dockpreps charge assignment when running dockprep.")) 
parser.add_argument("-t", "--timeprod", type = float, default= 0.2, help="Production simulation time in ns. Accepts floats and ints")
parser.add_argument("-dt", "--timestep", type = int, default= 4, help="Simulation timestep in fs. Accepts ints")
parser.add_argument("--resname", type = str, action="append", default= [], help="Residue names in PDB for which you want further analysis, eg ligand.\n"
                                                                            "several --resnames can be used at once \n if not specified all ligands added with --lig will get analyzed", required=False)
parser.add_argument("--debug", type = bool, default= False, help="debug mode, prints more information while running", required=False)
parser.add_argument("--dist", type = str, action="append", default= [], help="""a pair of atom numbers eg "resid 131 atom OG1", "resname UNK atom N1x" for which you want 
a distance plot eg for monitoring near-attack conformations. specify using MDAnalysis/VMD natural language queries""", required=False)

# Parse arguments
args = parser.parse_args()

# Extract into variables
pdb_file = args.pdb
pdb_fixer = args.pdbfix
ligand_files = args.lig
simulation_time_ns = args.timeprod
timestep = args.timestep
pdb_name = os.path.splitext(pdb_file)[0]
analysis_resnames = args.resname
debug = args.debug
dist_residues = args.dist

#calculate simulation length
production_steps = int(simulation_time_ns / (timestep * 10**-6))
equillibration_time = 40 #picoseconds
equillibration_steps = int(equillibration_time / (timestep * 10**-3))
reporting_time = 1 #ps
reporting_frequency = int(reporting_time / (timestep * 10**-3))

#this specifies the size, start and end of the 2D RMSD matrix
#BEWARE N^2 scaling operation
sparsity = int((simulation_time_ns * 1000 / reporting_time) / 500) #500 is the biggest amount of frames judged to be plausible to compute
if sparsity == 0:
    sparsity = 1
start_frame = 0
end_frame = -1 #last frame

#TODO figure out if we want HMR, constrain HBOND, 4 fs, langevinmiddle or another solution like 3 fs no HMR or a variable time langevin integrator (this would however lose the BAOB correction)
#TODO This is really important
#forcefield kwargs
forcefield_kwargs = {'constraints': HBonds, 'rigidWater': True, 'removeCMMotion': True, 'hydrogenMass' : 1.5 * amu }

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except:
    script_dir = os.getcwd()

cache_file = f"{script_dir}/ligands.json"


if pdb_fixer == "True" or pdb_fixer == "true":
    #remove hydrogens
    utils.remove_hydrogens(f"{pdb_name}.pdb", f"{pdb_name}_fixed.pdb")

    #Run PDBfixer
    fixer = PDBFixer(filename=f"{pdb_name}_fixed.pdb")
    #we remove and then re-add hydrogens to prevent shenanigans related to disulfide bonds
    fixer.addMissingHydrogens(7.4)  # add missing hydrogens
    fixer.findMissingResidues()

    #to avoid changing list we iterate over we copy everything to separate objects
    chains = list(fixer.topology.chains())
    keys = list(fixer.missingResidues.keys())  

    #Then we remove terminal residues we don´t know anything about
    for key in keys:
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    #fixer.removeHeterogens(False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    PDBFile.writeFile(fixer.topology, fixer.positions, open(f"{pdb_name}_fixed.pdb", 'w'))

    
    pdb = PDBFile(f"{pdb_name}_fixed.pdb")

elif pdb_fixer == "False" or pdb_fixer == "false":
    #if not fixing PDB
    pdb = PDBFile(f"{pdb_name}.pdb")

#if simulating with ligand
if ligand_files is not None:
    
    unnamed_ligands = 0
    ligand_mol = []
    lig_resnames = []

    for lig in ligand_files:
        #read ligand file
        ligand = Molecule.from_file(lig)

        #read name ensuring uppercase
        lig_name = os.path.splitext(os.path.basename(lig))[0].upper()
        
        # Automatically set as resname if it's exactly 3 letters or numbers long and no explicit resname is provided
        if len(re.findall(r'[A-Z0-9]', lig_name)) == 3:
            ligand_name = lig_name
        else:
            ligand_name = (f"UN{unnamed_ligands}")
            unnamed_ligands += 1

        #set name for atoms in residue to get desired behaviour from OFFtoolkit
        for atom in ligand.atoms:
            atom.metadata['residue_name'] = ligand_name
        
        #keep track of what ligands we are handling
        lig_resnames.append(ligand_name) 

        #add to list which will be added to topology
        ligand_mol.append(ligand)

    #if analysis not specified, analyze all added residues
    if analysis_resnames is None:
        analysis_resnames = lig_resnames

    # Specify the forcefield
    # Initialize a SystemGenerator using the Sage.2.1 for the ligand and tip3p for the water.
    system_generator = SystemGenerator(
        forcefields=['amber14-all.xml', 'amber14/tip3pfb.xml'],
        small_molecule_forcefield='openff-2.2.1.offxml',
        molecules=ligand_mol,
        forcefield_kwargs=forcefield_kwargs, cache=cache_file)
    
    #start a modeller
    modeller = Modeller(pdb.topology, pdb.positions)
    #residues=modeller.addHydrogens(system_generator.forcefield, pH = 7)
    
    # Adding ligand(s) to protein PDB
    existing_resnames = {res.name for res in pdb.topology.residues()}

    #add ligands to topology and save names
    for ligand in ligand_mol:
        if ligand.name not in existing_resnames:
            lig_top = ligand.to_topology()
            modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())
    
#if not simulating with ligand
elif ligand_files == None:
    # Specify the forcefield
    # Initialize a SystemGenerator using the Sage.2.1 for the ligand and tip3p for the water.
    
    system_generator = SystemGenerator(
        forcefields=['amber14-all.xml', 'amber14/tip3pfb.xml'],
        small_molecule_forcefield='openff-2.2.1.offxml',
        forcefield_kwargs=forcefield_kwargs, cache=cache_file)
    
    #start a modeller
    modeller = Modeller(pdb.topology, pdb.positions)
    #residues=modeller.addHydrogens(system_generator.forcefield, pH = 7)

modeller.deleteWater()
modeller.addSolvent(system_generator.forcefield, padding=1.0*nanometer)

# Create the system using the SystemGenerator
system = system_generator.create_system(modeller.topology)

#set precision and platform
properties = {"Precision": "mixed"} #improves energy conservation resulting in larger stable timesteps. decreases speed ca 5%
platform = Platform.getPlatformByName("CUDA")

#set up simulation
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, timestep * femtoseconds)
simulation = Simulation(modeller.topology, system, integrator, platform = platform)
simulation.context.setPositions(modeller.positions)

print("Minimizing energy")
simulation.minimizeEnergy()

print("Running NVT equillibration")
simulation.step(equillibration_steps)

system.addForce(openmm.MonteCarloBarostat(1 * bar, 300 * kelvin))
simulation.context.reinitialize(preserveState=True) #needed to add in the barostat

print("Running NPT equillibration")
simulation.step(equillibration_steps)

#save pdb
state = simulation.context.getState(getPositions=True)
with open(pdb_name + "_equillibrated.pdb", "w") as file:
    PDBFile.writeFile(simulation.topology, state.getPositions(), file)

#add reporters
#print to terminal
simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, volume=True, remainingTime=True, totalSteps=equillibration_steps *2 + production_steps, speed=True))

#saved to file
simulation.reporters.append(StateDataReporter(f"{pdb_name}_md_log.txt", reporting_frequency, step=True,
        potentialEnergy=True, temperature=True, volume=True))
simulation.reporters.append(DCDReporter(f"{pdb_name}_trajectory.dcd", reporting_frequency))

print("Running production NPT")
simulation.step(production_steps)

#save pdb
state = simulation.context.getState(getPositions=True)
with open(f"{pdb_name}_final.pdb", "w") as file:
    PDBFile.writeFile(simulation.topology, state.getPositions(), file)

import numpy as np
import MDAnalysis as mda 
import pandas as pd
import multiprocessing as mp
import time
import os

#only needed for devwork
import importlib
importlib.reload(utils)
importlib.reload(analysis)
importlib.reload(plot)

#TODO write a analysis class using the multithreaded functions
#TODO divide up this code into a analysis_main module

#get how much we can parallelize
n_jobs = mp.cpu_count()
align = True

#settings for the 2D RMSD algorithm
#Now set automatically above if not explicitlly requested
if False:
    sparsity = 40 #every nth frame will be analyzed in the 2D RMSD computation
    start_frame = None
    end_frame = None

print("Centering and aligning protein in PBC" if align else "Centering protein in PBC")

start_time = time.time()

#center proteins, This also resets the time of the first snapshot to 0 ps thus removing equillibration time
#TODO This now uses MDAnalysis bond guesser for the trajectory wrapping. This is suboptimal but i havent figured out how to communicate openmm topology
#to MDAnalysis except via ParMed which can´t handle rigid water 
centered_traj_name = f"{pdb_name}_trajectory.dcd"
final_pdb = f"{pdb_name}_final.pdb"
mda_traj = utils.parallel_center_trajectory(final_pdb, f"{pdb_name}_trajectory.dcd", align=align, n_jobs=n_jobs, output_filename = centered_traj_name) 

#rewrite final PDB after alignment
mda_traj.trajectory[-1]
mda_traj.atoms.write(final_pdb)

print(f"{round(time.time() - start_time,2)}s used for centering")

#dt is in ps
time_offset = 0 #redundant due to centering
mda_traj = mda.Universe(final_pdb, centered_traj_name, dt=reporting_time, time_offset=time_offset)


#load state data reporter information from memory
column_names = ["Step", "Potential Energy (kJ/mole)", "Temperature (K)", "Box Volume (nm^3)"]
#header=None prevents using the first row as headers avoiding "#"steps" as a name
#nrows = None allows us to read the whole csv not just the first 100 rows
md_log_df = pd.read_csv(f"{pdb_name}_md_log.txt", names=column_names, comment="#", header=None, nrows=None)  
#TODO put df in a whatcat.analysis class

# Extract Trajectory Time
times = np.array([ts.time for ts in mda_traj.trajectory])  # Extract time (in ps)
md_log_df.insert(0, "Time (ps)", times)

#plot reporter info from OPENMM
plot.line_plotter_2d(md_log_df["Time (ps)"], md_log_df["Potential Energy (kJ/mole)"], "Time (ps)", "Potential Energy (kJ/mole)", pdb_name, "energy")
plot.line_plotter_2d(md_log_df["Time (ps)"], md_log_df["Temperature (K)"], "Time (ps)", "Temperature (K)", pdb_name, "temperature")
plot.line_plotter_2d(md_log_df["Time (ps)"], md_log_df["Box Volume (nm^3)"], "Time (ps)", "Box Volume (nm^3)", pdb_name, "volume")

print("computing RMSD")
start_time = time.time()
# overwrite selection and stuff from centering
ref = mda_traj.copy()   # Create a copy of the universe in the first frame
ref.trajectory[0] #set frame to 0 explicitlly
backbone = mda_traj.select_atoms("backbone")

#create a temporary sparse trajectory for 2D RMSD analysis to save resorces in the N^2 scaling operation
sparse_traj = f"{pdb_name}_trajectory_sparse.dcd"
utils.write_trajectory(mda_traj, f"{pdb_name}_trajectory_sparse.dcd", sparsity=sparsity, start_frame=start_frame, end_frame=end_frame)

rmsd_analysis = mda.analysis.rms.RMSD(backbone, ref, select="backbone", ref_frame=0, superposition = False).run(backend="multiprocessing", n_workers= n_jobs)
rmsd_backbone = rmsd_analysis.results.rmsd[:, 2]  # Extract RMSD values (column index 2)

#save the 1D RMSD data
rmsd_back_df = pd.DataFrame({"RMSD backbone": rmsd_backbone})
md_log_df = pd.concat([md_log_df, rmsd_back_df], axis=1) 

rmsd_backbone_matrix = analysis.parallel_2d_rmsd(final_pdb, sparse_traj, "backbone", n_jobs=n_jobs)
plot.heatmap(rmsd_backbone_matrix, "tims (ps)", "tims (ps)", "RMSD (Å)", f"2D RMSD for backbone", f"2d_rmsd_backbone", pdb_name, sparsity, start_frame)

#define variables here so they are availible for desigining how to plot
residue_rmsd = []
rmsd_residue_names = []

#run residue specific analysis and plot
if len(analysis_resnames) > 0:
    for resname in analysis_resnames:

        #compute 1D RMSD
        residue = mda_traj.select_atoms(f"resname {resname}")
        rmsd_residue_names.append(f"RMSD {resname}")
        rmsd_analysis = mda.analysis.rms.RMSD(residue, ref, select=f"resname {resname}", ref_frame=0, superposition = False).run(backend="multiprocessing", n_workers= n_jobs)
        residue_rmsd.append(rmsd_analysis.results.rmsd[:, 2]) #we add a extra set of [] to "transpose" the list

        #compute 2D RMSD
        rmsd_2d_residue = analysis.parallel_2d_rmsd(final_pdb, sparse_traj, f"resname {resname}", n_jobs=n_jobs)
        plot.heatmap(rmsd_2d_residue, "tims (ps)", "tims (ps)", "RMSD (Å)", f"2D RMSD for {resname}", f"2d_rmsd_{resname}", pdb_name, sparsity, start_frame)

    #save the 1d data    
    rmsd_lig_df = pd.DataFrame(np.array(residue_rmsd).T, columns=rmsd_residue_names)
    md_log_df = pd.concat([md_log_df, rmsd_lig_df], axis=1) 
    plot.line_plotter_2d(md_log_df["Time (ps)"], md_log_df[["RMSD backbone"] + rmsd_residue_names], "Time (ps)", "RMSD (Å)", pdb_name, "1d_rmsd")

#if no extra analysis necessary, then we can plot right away
else:
    plot.line_plotter_2d(md_log_df["Time (ps)"], md_log_df["RMSD backbone"], "Time (ps)", "RMSD (Å)", pdb_name, "1d_rmsd")

#remove temp sparse traj
os.remove(sparse_traj)

print(f"{round(time.time() - start_time,2)}s used for RMSD")

#compute RMSF
print("computing RMSF")
start_time = time.time()

rmsf_result = analysis.calc_rmsf_parallel(final_pdb, centered_traj_name, n_jobs=n_jobs)

protein = mda_traj.select_atoms('protein')
residue_list = range(1, len(rmsf_result) +1)
plot.line_plotter_2d(residue_list, rmsf_result, "residue", "RMSF (Å)", pdb_name, "1d_rmsf")

print(f"{round(time.time() - start_time,2)}s used for RMSF")

# Compute Radius of Gyration
print("computing rgyr")
start_time = time.time()
protein = mda_traj.select_atoms('protein')
rga = mda.analysis.base.AnalysisFromFunction(analysis.radgyr, mda_traj.trajectory, protein, protein.masses, total_mass=np.sum(protein.masses)).run(backend="multiprocessing", n_workers= n_jobs)

rg_labels = ["Rg_all", "Rg_x", "Rg_y", "Rg_z"]
rg_df = pd.DataFrame(rga.results.timeseries, columns=rg_labels)
md_log_df = pd.concat([md_log_df, rg_df], axis=1)

plot.line_plotter_2d(md_log_df["Time (ps)"], md_log_df[rg_labels], "Time (ps)", "Radius of gyration (Å)", pdb_name, "rg")
print(f"{round(time.time() - start_time,2)}s used for rgyr")

if len(dist_residues) > 0:
    #compute distances
    print("computing pairwise distances")
    start_time = time.time()
    distances = analysis.calculate_pairwise_distances(final_pdb, centered_traj_name, dist_residues, backend="multiprocessing", n_workers=n_jobs)

    if debug == True:
        print("Shape of pairwise distance result:", distances.shape)  # Expected: (num_frames, num_pairs)
        print(distances)

    distance_labels = [i for i in dist_residues]
    distance_df = pd.DataFrame(distances, columns=distance_labels)
    md_log_df = pd.concat([md_log_df, distance_df], axis=1)

    print(f"{round(time.time() - start_time,2)}s used for pairwise distances")
    plot.line_plotter_2d(md_log_df["Time (ps)"], md_log_df[distance_labels], "Time (ps)", "Distance (Å)", pdb_name, "distances")

#save df
md_log_df.to_csv(f"{pdb_name}_results.csv", index=False)  # Saves without the index column

if debug == True:
    print(md_log_df)
    print(md_log_df.shape)
    print("\n")

#check equillibration
analysis.equillibration_check(md_log_df, reporting_time, dt=100)
