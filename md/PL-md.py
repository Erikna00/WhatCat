from openmm.app import *
from openmm import *
from openmm.unit import *

from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer

import argparse
import numpy as np
import scipy
import pandas as pd
import MDAnalysis as mda
import mdtraj as mdtraj

import prolif
from rdkit import DataStructs


from utils import utils, analysis, plot

import sys
import os
import multiprocessing as mp
import time
import re
import warnings
from openmm.app.metadynamics import Metadynamics
# suppress some MDAnalysis warnings when writing PDB files as well as the DCD timestep warning
warnings.filterwarnings('ignore')
#filter biopython warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="Bio.Application")


class Whatcat_md_runner():
    def __init__(self, 
                 pdb_file, ligand_files = None, restart = False,  platform="CUDA",
                 pdb_fixer=2, charge_correct = True, solvate = 2, ph = 7.4,
                 simulation_time_ns=None, timestep=4, reporting_time=10, equillibration_time=50,
                 debug = False ):
        """
        Creates a Whatcat_md object from python arguments.
        A Whatcat_md object can also be created via init_from_parse_args()

        pdb_file - path to pdbfile, must not contain any small molecules
        ligand - list of paths to ligand file, must be sdf if charge_correct = False, charge_correct converts each ligand with openbabel
        restart - bool for if to restart from _restart.xml files and _final.pdb printed by a previous run. Also appends to existing reporter path
                If restart is set most other parameters will go unused as the class will jump straight to simulation creation

        pdb_fixer - int [0,1,2] for wheter to run the pdbfixer script.
            0 - use pdb as is, user is totally responsible for pdb being valid
            1 - take pdb, pdbfix it retaining hydrogens (good for metals, bad for disulfides)
            2 - deprotonates pdb and the fixes it, adding new hydrogens appropriate for ph
        charge_correct - path to ligand file, converted to sdf and charges are added which lets system_generator know how many hydrogens to add
        solvate - int [0,1,2] for wheter to protonate system
            0 - use pdb as is without adding more solvent
            1 - retain existing (crystal) waters and add a box
            2 - remove existing water and add the water box
        ph - The pH for which protonation shall be suitable
        
        simulation_time_ns - production NPT simulation length in ns
        timestep - timestep in fs
        reporting_time - how often to save to DCD reporter in ps
        equillibration_time - how long to equillibrate for in ps
        
        analysis_resnames - internal variable set to all small molecule components
        """
        #TODO should __init__ have defaults? currentlly we can initillize values from all functions anyway
        
        #Extract into self.varibles
        self.pdb_file = pdb_file
        self.ligand_files = ligand_files
        self.restart = restart
        self.platform = platform

        self.pdb_fixer = pdb_fixer
        self.charge_correct = charge_correct
        self.solvate = solvate
        self.ph = ph

        self.simulation_time_ns = simulation_time_ns
        self.timestep = timestep #fs
        self.reporting_time = reporting_time #ps
        self.equillibration_time = equillibration_time #ps
        
        self.analysis_resnames = []

        self.debug = debug

        self.pdb_name = os.path.splitext(pdb_file)[0]
        self.ran_time = 0 *picoseconds

        try:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        except:
            self.script_dir = os.getcwd()


        #Restart overrides settings to allow loading of pdb_final
        if restart == True:
            self.pdb_name = self.pdb_name.replace("_final", "")

    def parse_set_default(self, attr_name, value):
        """
        Parses method variables compared to class variables to allow
        methods to use both class and set variables
        valid call is self.parse_set_default("charge_correct", charge_correct)
        """
        current_value = getattr(self, attr_name)
        if value is None:
            return current_value
        else:
            setattr(self, attr_name, value)
            return value

    def fix_pdb(self, pdb_fixer=None):
        """
        Converts a PDBfile to a openmm PDB object on all settings.
        Uses self.pdbfixer as the default setting but this can be overridden via the passed args
        if so self.pdbfixer is revised to the new value

        pdb_fixer - int [0,1,2] for wheter to run the pdbfixer script.
            0 - use pdb as is, user is totally responsible for pdb being valid
            1 - take pdb, pdbfix it retaining hydrogens (good for metals, bad for disulfides)
            2 - deprotonates pdb and the fixes it, adding new hydrogens appropriate for ph

        Saves a openmm PDBfile object as a class variable
        """
        pdb_fixer = self.parse_set_default("pdb_fixer", pdb_fixer) 

        if pdb_fixer == 1 or pdb_fixer == 2:

            if pdb_fixer == 2:
                #we remove and then re-add hydrogens to prevent shenanigans related to disulfide bonds
                utils.remove_hydrogens(self.pdb_file, f"{self.pdb_name}_fixed.pdb")

                #Run PDBfixer
                fixer = PDBFixer(filename=f"{self.pdb_name}_fixed.pdb")
            
            elif pdb_fixer == 1:
                #Retain hydrogens and fix anyway
                fixer = PDBFixer(filename=self.pdb_file)

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

            # add missing hydrogens after adding missing atoms
            fixer.addMissingHydrogens(self.ph)  

            PDBFile.writeFile(fixer.topology, fixer.positions, open(f"{self.pdb_name}_fixed.pdb", 'w'))

            
            pdb = PDBFile(f"{self.pdb_name}_fixed.pdb")

        elif pdb_fixer == 0:
            #if not fixing PDB
            pdb = PDBFile(self.pdb_file)

        else:
            raise ValueError("illegal option choosen for pdbfixer, valid options are 0, 1, 2")
        
        self.pdb = pdb
        return pdb
    
    def create_openmm_system(self, pdb=None, charge_correct=None, ph=None, solvate=None):
        """
        Creates a openmm_system and modeller object
        
        setting variabel values here changes the corresponding class variabel.
        Using = None uses the class variable

        returns system, modeller
        """
        #parse the inputs
        pdb = self.parse_set_default("pdb", pdb)
        charge_correct = self.parse_set_default("charge_correct", charge_correct)
        ph = self.parse_set_default("ph", ph)
        solvate = self.parse_set_default("solvate", solvate)
        
        #TODO add accellerated MD without colvars or metadynamics with colvars 
        #forcefield kwargs
        forcefield_kwargs = {'constraints': HBonds, 'rigidWater': True, 'removeCMMotion': True, 'hydrogenMass' : 1.5 * amu }

        #if simulating with ligand
        if len(self.ligand_files) > 0:
            cache_file = f"{self.script_dir}/ligands.json"
            unnamed_ligands = 0
            ligand_mol = []
            lig_resnames = []

            #start a cache finder object
            cache_finder = utils.FF_cache_reader(cache_file=cache_file)

            for lig in self.ligand_files:
                
                if charge_correct == True:
                    #TODO interface openeye as the primary pKa engine using either pkatyper or openff-toolkit.enumerate_protomers
                    #https://docs.eyesopen.com/toolkits/python/quacpactk/pkatypertheory.html
                    lig = utils.prepare_ligand_md(lig, ph)

                #read ligand file
                ligand = Molecule.from_file(lig)

                if cache_finder.check_molecule_in_cache(lig):
                    #This try-except was added due to GGPP -3 crashing during parametrization
                    #TODO this is bad but is needed for highlly charged ligands see https://github.com/openforcefield/openff-toolkit/issues/1741 https://github.com/openforcefield/openff-toolkit/issues/1911
                    #TODO Maybe wait for resolution of pull requests here? Issue likelly is #1911 with SCF not converging for GGPP -3?
                    #Alt use Psi4 OpenFF Recharge to interface Psi4 or something?
                    #Alt get openeye https://docs.openforcefield.org/projects/toolkit/en/latest/api/generated/openff.toolkit.topology.Molecule.html#openff.toolkit.topology.Molecule.assign_partial_charges
                    try:
                        print("assigning charges")
                        ligand.assign_partial_charges("am1bcc")
                        print("charges assigned")
                    except:
                        print("\n WARNING\nam1bcc failed, falling back to gasteiger charges\n")
                        ligand.assign_partial_charges("gasteiger")
                        print("charges assigned")

                #read name ensuring uppercase
                lig_name = os.path.splitext(os.path.basename(lig))[0].upper()
                
                # Automatically set as resname if it's exactly 3 letters or numbers long and no explicit resname is provided
                if len(re.findall(r'[A-Z0-9]', lig_name)) == 3:
                    ligand.name = lig_name
                else:
                    ligand.name = (f"UN{unnamed_ligands}")
                    unnamed_ligands += 1

                #set name for atoms in residue to get desired behaviour from OFFtoolkit
                for atom in ligand.atoms:
                    atom.metadata['residue_name'] = ligand.name
                
                #keep track of what ligands we are handling
                lig_resnames.append(ligand.name) 

                #add to list which will be added to topology
                ligand_mol.append(ligand)


            # Specify the forcefield
            # Initialize a SystemGenerator using the Sage.2.1 for the ligand and tip3p for the water.
            system_generator = SystemGenerator(
                forcefields=['amber14-all.xml', 'amber14/tip3pfb.xml'],
                small_molecule_forcefield='openff-2.2.1.offxml',
                molecules=ligand_mol,
                forcefield_kwargs=forcefield_kwargs, cache=cache_file)
            
            #start a modeller
            modeller = Modeller(pdb.topology, pdb.positions)

            #add ligands to topology, Ligand already in PDB not supported
            for ligand in ligand_mol:
                lig_top = ligand.to_topology()
                modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())

            #save all ligands for analysis
            for ligand_name in lig_resnames:
                self.analysis_resnames.append(f"resname {ligand_name}")
            
        #if not simulating with ligand
        elif len(self.ligand_files) == 0:
            # Specify the forcefield
            # Initialize a SystemGenerator using the Sage.2.1 for the ligand and tip3p for the water.
            
            system_generator = SystemGenerator(
                forcefields=['amber14-all.xml', 'amber14/tip3pfb.xml'],
                small_molecule_forcefield='openff-2.2.1.offxml',
                forcefield_kwargs=forcefield_kwargs)
            
            #start a modeller
            modeller = Modeller(pdb.topology, pdb.positions)

        if solvate == 2:
            #remove all water
            modeller.deleteWater()
        if solvate > 0:
            #add solvent box
            modeller.addSolvent(system_generator.forcefield, padding=1.0*nanometer)

        # Create the system using the SystemGenerator
        system = system_generator.create_system(modeller.topology)
        self.system = system
        self.modeller = modeller

        return system, modeller
    
    def create_openmm_simulation(self, system=None, modeller=None, timestep = None):
        """
        Creates and returns a openmm simulation object ready for use with NVT
        """
        #Parse inputs
        system = self.parse_set_default("system", system)
        modeller = self.parse_set_default("modeller", modeller)
        timestep = self.parse_set_default("timestep", timestep)

        #set precision and platform
        if self.platform == "OPENCL" or self.platform == "CUDA":
            properties = {"Precision": "mixed"} #improves energy conservation resulting in larger stable timesteps. decreases speed ca 5%
            platform = Platform.getPlatformByName(self.platform)

        #set up simulation
        self.integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, timestep * femtoseconds)
        simulation = Simulation(modeller.topology, system, self.integrator, platform = platform)
        simulation.context.setPositions(modeller.positions)

        self.simulation = simulation

        return simulation
    
    def restart_simulation_from_file(self, pdb = None, restart_pdb_file=None): 
        """
        Reads xml restart files and restarts a simulation object from the same.
        If pdb_file is set, self.pdbfile is overwritten and self.name is set to the restart_pdb_file with _final removed
        else self.pdb_name is inspected and _final is removed if present
        """

        if restart_pdb_file is not None:
            self.pdb_file = restart_pdb_file
            #remove restart if present
            self.pdb_name = restart_pdb_file.replace("_final", "")
        
        self.pdb_name = self.pdb_name.replace("_final", "")
        
        #set file basename of all restart files
        checkpoint_filebase = f"{self.pdb_name}_restart"

        pdb = PDBFile(f"{self.pdb_name}_final.pdb")
        self.pdb = pdb

        #Read in XML:d data
        with open(f"{checkpoint_filebase}_system.xml", "r") as f:
            self.system = XmlSerializer.deserialize(f.read())

        with open(f"{checkpoint_filebase}_integrator.xml", "r") as f:
            self.integrator = XmlSerializer.deserialize(f.read())

        simulation = Simulation(pdb.topology, self.system, self.integrator)
        simulation.loadState(f"{checkpoint_filebase}_state.xml")

        self.simulation = simulation
        self.timestep = int(simulation.integrator.getStepSize().value_in_unit(femtosecond))
        print(f"\nSimulation restarted with stepsize of {self.timestep} fs")

        self.simulation = simulation
        self.ran_time = simulation.context.getState().getTime()

        return simulation
    
    def equillibrate_simulation(self, simulation = None, equillibration_time=None):
        """
        Takes a openmm simulation object and equillibrates it for the provided time converting it
        to a NPT simulation in the process.
        Prints equillibration step log to md_log_equil.txt
        """

        #TODO maybe separate NVT and NPT equillibration time?
        simulation = self.parse_set_default("simulation", simulation)
        equillibration_time = self.parse_set_default("equillibration_time", equillibration_time)

        equillibration_steps = int(equillibration_time / (self.timestep * 10**-3))

        print("Minimizing energy")
        simulation.minimizeEnergy()

        #add equillibration reporter
        reporting_frequency = int(self.reporting_time / (self.timestep * 10**-3))
        simulation.reporters.append(StateDataReporter(f"{self.pdb_name}_md_log_equil.txt", reporting_frequency, step=True,
        potentialEnergy=True, temperature=True, volume=True, append = self.restart))
        simulation.reporters.append(DCDReporter(f"{self.pdb_name}_trajectory_equil.dcd", reporting_frequency, append = self.restart))

        print("Running NVT equillibration")
        simulation.step(equillibration_steps)

        self.system.addForce(openmm.MonteCarloBarostat(1 * bar, 300 * kelvin))
        simulation.context.reinitialize(preserveState=True) #needed to add in the barostat

        print("Running NPT equillibration")
        simulation.step(equillibration_steps)

        #save pdb
        state = simulation.context.getState(getPositions=True)
        with open(self.pdb_name + "_equillibrated.pdb", "w") as file:
            PDBFile.writeFile(simulation.topology, state.getPositions(), file)

        #reset simulation time to 0 for decent analysis
        simulation.currentStep = 0  # Reset step counter
        simulation.context.setTime(0 * picoseconds)  # Reset simulation time to 0 ps

        #remove equillibration reporter
        simulation.reporters.clear()

        self.simulation = simulation
        self.equillbration_steps = equillibration_steps

        return simulation
    
    def add_metadynamics(self, atom_indices, min_value=0.0, max_value=2.0, bias_factor=10.0, hill_height=1.0, hill_width=0.1, hill_frequency=500, grid_width=100):
        """
        Adds a metadynamics bias to the system using OpenMM's Metadynamics class.
        Also creates the collective variable (CV) force.

        Parameters:
            atom_indices: list
                Atom indices for the CV. For "bond", provide [i, j]. For "angle", provide [i, j, k].
            min_value: float
                Minimum value of the CV (in CV units).
            max_value: float
                Maximum value of the CV (in CV units).
            bias_factor: float
                Bias factor for well-tempered metadynamics.
            hill_height: float
                Height of the deposited hills (in kJ/mol).
            hill_width: float
                Width (sigma) of the hills (in CV units).
            hill_frequency: int
                How often (in steps) to deposit a hill.
            grid_width: int
                Number of bins for the bias grid.

        Returns:
            metadynamics: openmm.app.metadynamics.Metadynamics
                The metadynamics object (stores bias and can be used for analysis).
            cv_force: openmm.CustomCVForce
                The collective variable force object.
        """

        if not hasattr(self, "simulation"):
            raise RuntimeError("Simulation must be created before adding metadynamics.")

        # Create the CV force using PBC
        if len(atom_indices) == 2:
            # Harmonic bond CV
            cv = CustomBondForce("r")
            cv.addBond(int(atom_indices[0]), int(atom_indices[1]), [])
            cv_force = CustomCVForce("bond")
            cv_force.addCollectiveVariable("bond", cv)
        elif len(atom_indices) == 3:
            # Harmonic angle CV
            cv = CustomAngleForce("theta")
            cv.addAngle(int(atom_indices[0]), int(atom_indices[1]), int(atom_indices[2]), [])
            cv_force = CustomCVForce("angle")
            cv_force.addCollectiveVariable("angle", cv)
        else:
            raise ValueError("cv_type must be 'bond' or 'angle'.")

        # Add the CV force to the system
        self.simulation.system.addForce(cv_force)

        # Set up the bias variable grid
        grid_min = [min_value]
        grid_max = [max_value]
        grid_bins = [grid_width]

        # Create the Metadynamics object
        meta = Metadynamics(
            system=self.simulation.system,
            collectiveVariables=[cv_force],
            temperature=300*kelvin,
            biasFactor=bias_factor,
            height=hill_height*kilojoule_per_mole,
            frequency=hill_frequency,
            biasDir=None,
            saveFrequency=0,
            gridMin=grid_min,
            gridMax=grid_max,
            gridWidth=[hill_width],
            gridBins=grid_bins,
            wellTempered=True
        )

        self.metadynamics = meta
        return meta, cv_force

    def run_prod_simulation(self, simulation = None, simulation_time_ns=None, reporting_time = None):
        """
        Runs the production NPT simulation for the set amount of time.
        """
        simulation = self.parse_set_default("simulation", simulation)
        simulation_time_ns = self.parse_set_default("simulation_time_ns", simulation_time_ns)
        reporting_time = self.parse_set_default("reporting_time", reporting_time)

        #calculate simulation length
        production_steps = int(simulation_time_ns / (self.timestep * 10**-6))
        reporting_frequency = int(reporting_time / (self.timestep * 10**-3))

        #add reporters
        #print to terminal
        simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
                potentialEnergy=True, temperature=True, volume=True, remainingTime=True, totalSteps= production_steps, speed=True))

        #saved to file
        simulation.reporters.append(StateDataReporter(f"{self.pdb_name}_md_log.txt", reporting_frequency, step=True,
                potentialEnergy=True, temperature=True, volume=True, append = self.restart))
        simulation.reporters.append(DCDReporter(f"{self.pdb_name}_trajectory.dcd", reporting_frequency, append = self.restart))

        print("Running production NPT")
        simulation.step(production_steps)

        #save pdb
        state = simulation.context.getState(getPositions=True)
        with open(f"{self.pdb_name}_final.pdb", "w") as file:
            PDBFile.writeFile(simulation.topology, state.getPositions(), file)
        
        #save checkpoints of state, system and integrator
        checkpoint_filebase = f"{self.pdb_name}_restart"
        simulation.saveState(f"{checkpoint_filebase}_state.xml")

        with open(f"{checkpoint_filebase}_system.xml", 'w') as file:
            file.write(XmlSerializer.serialize(self.system))
        with open(f"{checkpoint_filebase}_integrator.xml", 'w') as file:
            file.write(XmlSerializer.serialize(self.integrator))
        
        self.simulation = simulation
        return simulation
    
    def create_analysis(self):
        """
        Creates a Whatcat_md_analysis object based on the simulation and returns it
        Exports pdb_name, simulation.topology, trajectory filename, reporting time and simulation time to analyzer
        """

        md_analysis = Whatcat_md_analysis(self.pdb_name, self.simulation.topology, f"{self.pdb_name}_trajectory.dcd", self.reporting_time, self.simulation_time_ns)

        return md_analysis


class Whatcat_md_analysis:
    
    def __init__(self, basename, topology, traj_file, reporting_time, simulation_time_ns, align = True, plot = True, start_time=0):
        """
        This class analyzes MD simulations by wrapping MDAnalysis in a parallelized executor using
        divide and conquer methodologies when the MDAnalysis function does not have native parallelization

        Parameters
            topology : str or OPENMM topology compatible with MDAnalysis 
                A PDB file or OPENMM topology. Needs to contain bonding information for center_align_traj to work
                This is fulfilled by passing simulation.topology from openmm
            traj_file : str
                A trajectory file such as .dcd
            self.basename : str
                basename for file output
            align : Bool 
                if you want trajectory aligned to first frame
            plot : 
                Bool if you want plots or only save to df
        
        Returns
            A whatcat_md_analysis object
        """
        self.traj_file = traj_file
        self.topology = topology
        self.basename = basename
        self.align = align
        self.plot = plot

        self.reporting_time = reporting_time
        self.simulation_time_ns = simulation_time_ns
        
        self.sparse_traj = None

        self.time_df = pd.DataFrame()
        self.residue_df = pd.DataFrame()

        #get how much we can parallelize
        self.n_jobs = mp.cpu_count()

    def read_md_log(self, md_log = None):
        """
        Reads the log file from whatcat_md_runner into self.time_df

        Parameters
            md_log : str
                Path to a log file. default corresponds to basename_md_log.txt which works for files from whatcat_md_runner
                if None, f"{self.basename}_md_log.txt"

        Returns
            np.ndarray
                2D NumPy array of shape (num_frames, num_pairs) where each row contains the
                computed distances for the specified pairs in that frame.
            also adds columns to self.time_df
        """
        if md_log is None:
            md_log = f"{self.basename}_md_log.txt"

        #dt is in ps
        time_offset = 0 #redundant due to centering
        u = mda.Universe(self.topology, self.traj_file, dt=self.reporting_time, time_offset=time_offset)

        #load state data reporter information from memory
        column_names = ["Step", "Potential Energy (kJ/mole)", "Temperature (K)", "Box Volume (nm^3)"]
        
        #header=None prevents using the first row as headers avoiding "#"steps" as a name
        #nrows = None allows us to read the whole csv not just the first 100 rows
        self.time_df = pd.read_csv(f"{self.basename}_md_log.txt", names=column_names, comment="#", header=None, nrows=None)  

        # Extract Trajectory Time
        times = np.array([ts.time for ts in u.trajectory])  # Extract time (in ps)
        self.time_df.insert(0, "Time (ps)", times)

        #plot reporter info from OPENMM
        plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df["Potential Energy (kJ/mole)"], "Time (ps)", "Potential Energy (kJ/mole)", self.basename, "energy")
        plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df["Temperature (K)"], "Time (ps)", "Temperature (K)", self.basename, "temperature")
        plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df["Box Volume (nm^3)"], "Time (ps)", "Box Volume (nm^3)", self.basename, "volume")

    def center_align_traj(self, pdb_ending = "_final.pdb", topology_ending = "_trajectory.dcd", topology_pdb = None):
        """
        Centers the protein in the periodic box.
        Optionally aligns the protein to the first frame to remove tumbling
        Aligning is necessary for some of the other analyses in this class to work

        Parameters
            pdb_ending : str
                the fileending which should be added to self.basename to write the centered structure to
            topology_ending : str
                the fileending which should be added to self.basename to write the centered trajectory to
                defaults are set to overwrite the files from Whatcat_md_runner.
                self.traj_file will be set to self.basename + topology_ending
            
            topology_pdb : str
                The filepath to the PDB used as topology when converting the DCD back from using a MDA header to using a OPENMM header.
                This issue stems from the DCD format not being well defined and this is a hacky solution.
                If left as None, topology_pdb = f"{self.basename}_final.pdb" to conform with whatcats MD runner

        Returns
            Nothing and does not add to df
        """
        
        print("Centering and aligning protein in PBC" if self.align else "Centering protein in PBC")

        start_time = time.time()

        #center proteins, This also resets the time of the first snapshot to 0 ps thus removing equillibration time
        centered_traj_name = f"{self.basename}{topology_ending}"
        self.traj_file = centered_traj_name
        final_pdb = f"{self.basename}{pdb_ending}"

        mda_traj = utils.parallel_center_trajectory(self.topology, f"{self.basename}_trajectory.dcd", align=self.align, n_jobs=self.n_jobs, output_filename = centered_traj_name) 

        #rewrite final PDB after alignment
        mda_traj.trajectory[-1]
        mda_traj.atoms.write(final_pdb)

        if self.align == False:
            print("WARNING trajectory was not aligned. Subsequent analysis might be inaccurate")

        #convert the DCD back to OPENMM compatible format
        #DO NOT CHANGE OR MEDDEL WITH THIS
        #The issue is complex and related to the DCD header
        #Fixed in OPENMM https://github.com/openmm/openmm/pull/4899
        #TODO when OPENMM 8.0.3 is released
        if topology_pdb is None:
            topology_pdb = f"{self.basename}_final.pdb"

        md_traj = mdtraj.load(centered_traj_name, top=final_pdb)    
        md_traj.save_dcd(centered_traj_name)

        print(f"{round(time.time() - start_time,2)}s used for centering")

    def calc_pairwise_distances(self, analysis_distances):
        """
        Calculate pairwise distances between specified atom pairs for a trajectory.
        Each element of `atom_pairs` should be a string with two selection
        queries separated by a comma.
        
        Parameters
            pdb_file : str
                Path to the topology (PDB) file.
            traj_file : str
                Path to the trajectory file.
            atom_pairs : np.ndarray or list of str
                Array/list of strings specifying atom pairs (MDAnalysis selection language) in the format:
                "selection1, selection2"
                eg "resid 131 and name OG1, resname UNK and name N1x"
        
        Returns
            np.ndarray
                2D NumPy array of shape (num_frames, num_pairs) where each row contains the
                computed distances for the specified pairs in that frame.
            also adds columns to self.time_df
        """

        #if empty input
        if len(analysis_distances) == 0:
            return None
        
        #compute distances
        print("computing pairwise distances")
        start_time = time.time()

        # Convert each input string into a tuple of two selection strings.
        pair_selections = []
        for pair in analysis_distances:
            parts = pair.split(",")
            if len(parts) != 2:
                raise ValueError("Each atom pair must be in the format: 'selection1, selection2'")
            pair_selections.append((parts[0], parts[1]))
        
        # Create the MDAnalysis Universe.
        u = mda.Universe(self.topology, self.traj_file)
        
        # Set up AnalysisFromFunction.
        # Here, we pass u.trajectory as the trajectory to iterate over and u.atoms as the AtomGroup
        # to be updated on each frame. The 'pair_selections' tuple is passed as an argument to our function.
        work = mda.analysis.base.AnalysisFromFunction(analysis.compute_pairwise_distance_frame, u.trajectory, u.atoms, pair_selections)
        work.run(backend = "multiprocessing", n_workers = self.n_jobs)

        self.time_df = pd.concat([self.time_df, pd.DataFrame(work.results.timeseries, columns = analysis_distances)], axis = 1)

        if self.plot:
            plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df[analysis_distances], "Time (ps)", "Distance (Å)", self.basename, "distances")

        print(f"{round(time.time() - start_time,2)}s used for pairwise distances")

        return work.results.timeseries
    
    def calc_ca_rmsf(self, colored_pdb_ending = "_final.pdb"):
        """
        Compute RMSF with on-the-fly trajectory alignment using MDAnalysis transformations, 
        ensuring low memory usage and parallelization by processing frames in parallel.

        Parameters
            colored_pdb_ending : str
                During this analysis a PDB colored according to rmsf is written to basename + colored_pdb_ending
                Default reads in self.topology and overwrites the final pdb from whatcat_md_runner with the new information
            Selection : str
                The selection for which RMSF is calculated. No matter the selection, the alignment and averaging of structures 
                is done using "protein and name CA"

        Returns:
            np.ndarray: Computed RMSF values per residue.
        """
        #compute RMSF
        print("computing protein Ca RMSF")
        start_time = time.time()
        
        #dont change without looking into alignment and returned matrix size
        #If you want another RMSF you will have to write another function and adapt analysis.compute_rmsf_chunk
        selection = "protein and name CA and not resname ACE NME"

        # Load Universe once to compute the average structure.
        u = mda.Universe(self.topology, self.traj_file)
        
        # Compute the average structure (for alignment reference)
        avg_pdb=f"{self.basename}_avg_structure.pdb"
        avg_struct = mda.analysis.align.AverageStructure(u, u, select=selection, ref_frame=0, filename=avg_pdb).run()

        # Set unit cell dimensions from the first frame before writing
        avg_struct.results.universe.dimensions = u.trajectory[0].dimensions
        
        n_frames = u.trajectory.n_frames
        
        # Split frame indices evenly among workers.
        frame_chunks = np.array_split(range(n_frames), self.n_jobs)
        
        #TODO change parallelism scheme here
        with mp.Pool(self.n_jobs) as pool:
            results = pool.starmap(
                analysis.compute_rmsf_chunk,
                [(self.topology, self.traj_file, list(chunk), selection, avg_pdb)
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

        #add rmsf values to self.residue_df
        self.residue_df = pd.concat([self.residue_df, pd.DataFrame(rmsf_values, columns=["Ca RMSF"])], axis=1)
        
        # Save RMSF as B-factors in a PDB file.
        u_out = mda.Universe(self.topology, self.traj_file)
        u_out.add_TopologyAttr('tempfactors')
        protein = u_out.select_atoms(selection)
        
        # Assign computed RMSF values to residues (assuming one RMSF per CA atom)
        for residue, r_value in zip(protein.residues, rmsf_values):
            residue.atoms.tempfactors = r_value

        #write pdb 
        u_out.trajectory[0]
        u_out.atoms.write(f"{self.basename}{colored_pdb_ending}")

        #plot the RMSF values
        if self.plot:
            protein = u.select_atoms(selection)
            residue_list = range(1, len(rmsf_values) +1)
            plot.line_plotter_2d(residue_list, rmsf_values, "residue", "RMSF (Å)", self.basename, "1d_rmsf")

            print(f"{round(time.time() - start_time,2)}s used for RMSF")

        
        return rmsf_values
    
    def calc_1d_rmsd(self, selection_list = ["backbone"]):
        """ 
        Calculates RMSD over the trajectory for the given selections.
        Assumes traj is centered and aligned.
        
        Parameters:
            selection_list : list of str
                The selections for which RMSD is calculated

        Returns:
            Nothing, results are appended to self.time_df
        """
        print("computing 1D RMSD")
        start_time = time.time()

        u = mda.Universe(self.topology, self.traj_file)

        ref = u.copy()   # Create a copy of the universe in the first frame
        ref.trajectory[0] #set frame to 0 explicitlly

        for selection in selection_list:
            rmsd_analysis = mda.analysis.rms.RMSD(u, ref, select=selection, ref_frame=0, superposition = False).run(backend="multiprocessing", n_workers= self.n_jobs)
            rmsd_backbone = rmsd_analysis.results.rmsd[:, 2]  # Extract RMSD values (column index 2)

            #save the 1D RMSD data
            temp_df = pd.DataFrame({f"RMSD {selection}": rmsd_backbone})
            self.time_df = pd.concat([self.time_df, temp_df], axis=1) 

        #plot if plotting
        if self.plot:
            # Select all columns that start with "RMSD"
            rmsd_columns = [col for col in self.time_df.columns if col.startswith("RMSD")]

            # Ensure that at least one RMSD column is found
            if rmsd_columns:
                plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df[rmsd_columns], "Time (ps)", "RMSD (Å)", self.basename, "1d_rmsd")
                
            else:
                print("No RMSD columns found in the dataframe.")

        print(f"{round(time.time() - start_time,2)}s used for 1D RMSD")

    def calc_rgyr(self, selection = "protein", legend = None):
        """ 
        Calculates the mass weighted radgyr over the trajectory for the given selection.
        Assumes traj is centered and aligned.
        Does not actually plot anything as we want all RMSD:s in the same graph. use self.plot_rmsd()
        
        Parameters:
            selection : str
                The selection for which Rgyr is calculated
            legend : str
                The name you want to be added to the column names if the selection string is long
                column names are labeled as f"Rg_all_{selection}"
                if none legend = selection

        Returns:
            Nothing, result is appended to self.time_df
        """
        # Compute Radius of Gyration
        print("computing rgyr")
        start_time = time.time()

        if legend is None:
            legend = selection

        u = mda.Universe(self.topology, self.traj_file)

        atomgroup = u.select_atoms(selection)
        rga = mda.analysis.base.AnalysisFromFunction(analysis.radgyr, u.trajectory, atomgroup, atomgroup.masses, total_mass=np.sum(atomgroup.masses)).run(backend="multiprocessing", n_workers= self.n_jobs)

        rg_labels = [f"Rg_all_{legend}", f"Rg_x_{legend}", f"Rg_y_{legend}", f"Rg_z_{legend}"]
        rg_df = pd.DataFrame(rga.results.timeseries, columns=rg_labels)
        self.time_df = pd.concat([self.time_df, rg_df], axis=1)

        if self.plot:
            plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df[rg_labels], "Time (ps)", "Radius of gyration (Å)", self.basename, "rg")

        print(f"{round(time.time() - start_time,2)}s used for rgyr")

    
    def write_sparse_traj(self, start_frame = 0, end_frame = -1, max_frames = 500):
        """ 
        Writes a sparse trajectory for analysis.
        Useful for parallel_2d_rmsd as the scaling is N^2
        Sparsity is currentlly calculated so that the sparse traj will at most have max_frames frames
        Saves the sparse trajectory to self.basename_sparse.dcd
        
        Parameters:
            start_frame : int
                The frame at which to start printing the trajectory
            end_frame : int
                The frame at which to stop printing the trajectory
            max_frames : int
                The number of frames we want at most in the sparse traj

        Returns:
            str : name of the sparse trajectory
        """
        
        #calculate total frames in query 
        u = mda.Universe(self.topology, self.traj_file)
        tot_frames = u.trajectory.n_frames

        #add args to class variables to plot correctlly later on
        self.start_frame = start_frame
        self.end_frame = end_frame

        if end_frame == -1 or end_frame >= tot_frames -1 :
            end_frame = tot_frames -1

        #this specifies the size, start and end of the 2D RMSD matrix
        #BEWARE N^2 scaling operation
        sparsity = (tot_frames - (tot_frames - end_frame) - start_frame)/ max_frames #500 is the biggest amount of frames judged to be plausible to compute
        
        #Round up and convert to int by exploiting floating point remainder
        sparsity = int(sparsity // 1 + (sparsity % 1 > 0))

        #calc n_frames in sparse traj for printout to user
        frames_in_sparse = int((tot_frames - (tot_frames - end_frame) - start_frame)/ sparsity)

        #if we round down to 0 we round back up
        if sparsity == 0:
            sparsity = 1

        self.sparsity = sparsity
        
        self.sparse_traj = f"{self.basename}_trajectory_sparse.dcd"
        utils.write_trajectory(u, self.sparse_traj, sparsity=sparsity, start_frame=start_frame, end_frame=end_frame)

        print(f"\nwrote sparse traj from frame {start_frame} to {end_frame} with sparsity {sparsity} for a total of {frames_in_sparse} frames \n")

        return self.sparse_traj
    
    def remove_sparse_traj(self):
        """
        Removes the sparse traj printed for 2D RMSD analysis
        """

        #remove temp sparse traj
        os.remove(self.sparse_traj)
        print("removed sparse trajectory from disk")
        
    
    def calc_2d_rmsd(self, selection_list=["backbone"], legend_list = None):
        """ 
        Compute the full 2D RMSD matrix efficiently using process-based parallelism.
        runs write_sparse_traj() if self.sparse_traj is not set and then removes the sparse trajectory.
        If self.sparse_traj is set this function does not remove the sparse traj.
        
        Parameters:
            selection_list : list of str 
                Atom selection string for RMSD calculation.
            legend_list : list of str
                The name of the selection you want in the legend
                eg selection = "protein and name CA", legend = "Calpha"
                if legend_list = None or lengths dont match, legend_list = selection_list


        Returns:
            Tuple of legend_list and rmsd_matrix_dict
            rmsd_matrix_dict : dictionary
                Dict of legend, np.ndarray pairs: The ndarray is the computed symmetric RMSD matrix for a certain legend
                The dictionary indexes are the legend list, or if that is not given, the selection list stripped of "resname "
        """

        print(f"computing 2D RMSD")
        start_time = time.time()
        delete = False

        if self.sparse_traj is None:
            self.write_sparse_traj()
            delete = True
        
        u = mda.Universe(self.topology, self.sparse_traj)
        n_frames = u.trajectory.n_frames

        # Generate all (i, j) pairs for the upper triangle where j > i
        frame_pairs = [(i, j) for i in range(n_frames) for j in range(i + 1, n_frames)]
        split_pairs = np.array_split(frame_pairs, self.n_jobs)  # Distribute pairs across jobs

        #check that user input was valid
        if legend_list is None:
            legend_list = utils.strip_str_from_list(selection_list, "resname ")

        elif len(selection_list) != len(legend_list):
            raise ValueError("selection list did not match the legend list in 2D RMSD calculation")

        #start the results dict
        rmsd_matrix_dict = {}

        #iterate over calculations
        for selection, legend in zip(selection_list, legend_list):

            with mp.Pool(self.n_jobs) as pool:
                #TODO revise all parallelization here and in RMSF to start more reliablly
                results_list = pool.starmap(analysis.compute_2d_rmsd_block, [(self.topology, self.traj_file, selection, list(pairs)) for pairs in split_pairs])

            # Assemble the full symmetric RMSD matrix
            rmsd_matrix = np.zeros((n_frames, n_frames))

            for results in results_list:
                for (i, j), value in results.items():
                    rmsd_matrix[i, j] = value  # Upper triangle
                    rmsd_matrix[j, i] = value  # Mirror to lower triangle

            #write to file
            pd.DataFrame(rmsd_matrix).to_csv(f"{self.basename}_2d_rmsd_{legend}.csv")

            #add to dict
            rmsd_matrix_dict[f"{legend}"] = rmsd_matrix

            if self.plot:
                plot.heatmap(rmsd_matrix, "Time (ps)", "Time (ps)", "RMSD (Å)", f"2D RMSD for {legend}", f"2d_rmsd_{legend}", self.basename, self.reporting_time, self.sparsity, self.start_frame)

        #if the user did not start a sparse traj remove the automatically created one
        if delete:
            self.remove_sparse_traj()


        print(f"{round(time.time() - start_time,2)}s used for 2D RMSD")

        return legend_list, rmsd_matrix_dict
    
    def equillibration_check(self, dt = 100, r2cutoff = 0.05):
        """
        Uses a linear regression to find out if the simulation is properlly equillibrated

        Parameters:
            dt: int 
                for how many ps do you want to analyze if the simulation is equillibrated
            r2cutoff: float
                The cutoff for when a R^2 value no longer indicates a equillibrated simulation

        Returns:
            tuple : R^2 energy, R^2 volume, R^2 temp 
        """
        #Panda step 0 corresponds to the first step after equillibration
        analyzed_snaps = int((dt / self.reporting_time) +1) #how many reports are within 100 ps, +1 because noninclusive slicing
        step = self.time_df["Step"][0:analyzed_snaps]  # Produces a Pandas Series
        potential_energy = self.time_df["Potential Energy (kJ/mole)"][0:analyzed_snaps]
        temperature = self.time_df["Temperature (K)"][0:analyzed_snaps]
        volume = self.time_df["Box Volume (nm^3)"][0:analyzed_snaps]

        #calculate if system shows trends in some direction which indicates a too short equillibration
        regr_energy = scipy.stats.linregress(step, potential_energy)
        regr_volume = scipy.stats.linregress(step, volume)
        regr_temp = scipy.stats.linregress(step, temperature)

        if regr_energy.rvalue ** 2 < r2cutoff and regr_volume.rvalue ** 2 < r2cutoff and regr_temp.rvalue ** 2 < r2cutoff:
            print(f"\nR^2 for first 100 ps is good.\nenergy: {regr_energy.rvalue ** 2} \nvolume: {regr_volume.rvalue ** 2} \ntemp: {regr_temp.rvalue ** 2}\n")
        else:
            print(f"\nSIMULATION LIKELLY NOT EQUILLIBRATED.\nenergy: {regr_energy.rvalue ** 2} \nvolume: {regr_volume.rvalue ** 2} \ntemp: {regr_temp.rvalue ** 2}\n")
        
        return regr_energy.rvalue ** 2, regr_volume.rvalue ** 2, regr_temp.rvalue ** 2
    
    def run_prolif(self, analysis_resnames, analyze_water = False, start_ps = 0, stop_ps = -1, sparsity = 1):
        """
        Uses prolif and MDAnalysis to generate a interaction fingerprint and barcode

        Parameters:
            analysis_resnames: list
                list of selection strings for interaction analysis
                eg ["resname LIG"]
            analyze_water: bool
                Whether to include ligand interactions with water
            start_ps: int
                At what point in the traj shall we stat analysis?
            stop_ps: int
                At what point in the traj shall we stop analysis? -1 means we run the entire trajectory.
            sparsity: int
                How often do we sample the trajectory for snapshots to be analyzed?

        Returns:
            tuple : list of dictonary keys, dictionary of interaction dataframes
        """

        print("Computing interaction fingerprints")
        start_time = time.time()

        # load topology and trajectory
        u = mda.Universe(self.topology, self.traj_file)

        #create a dictionary of dataframes to store the interactions
        interaction_df_dict = {}
        
        for ligand_selector_string in analysis_resnames:
            #select the ligand
            ligand_selection = u.select_atoms(ligand_selector_string)

             # create selections for the protein (and water)
            if analyze_water:
                protein_selection = u.select_atoms("(protein or resname WAT) and byres around 20.0 group ligand",
                ligand=ligand_selection)

            else:
                protein_selection = u.select_atoms("protein and byres around 20.0 group ligand", ligand=ligand_selection)

            # create a molecule from the MDAnalysis selection
            ligand_mol = prolif.Molecule.from_mda(ligand_selection)

            # use default interactions
            fp = prolif.Fingerprint()

            #calculate frame starts and ends
            start_frame = round(start_ps/self.reporting_time)

            if stop_ps == -1:
                stop_frame = -1
            else:
                stop_frame = round(stop_ps/self.reporting_time)

            # run on a slice of the trajectory frames: from start to stop with a step of sparsity
            fp.run(u.trajectory[start_frame:stop_frame:sparsity], ligand_selection, protein_selection, n_jobs = self.n_jobs)

            #plot the barcode diagram
            ax = fp.plot_barcode()

            #Modify the x axis to be in time
            old_ticks = ax.get_xticks()                # returns list
            ps_per_frame = self.reporting_time * sparsity
            new_ticks = (old_ticks + start_frame) * ps_per_frame

            #check if ns or ps
            if max(new_ticks) > 1000:
                new_ticks = new_ticks / 1000

                #dont modify the data but relabel the X-axis and its tick marks
                ax.set_xticklabels([f"{t:.1f}" for t in new_ticks])
                ax.set_xlabel("Time (ns)")

            else: 
                #dont modify the data but relabel the X-axis and its tick marks
                ax.set_xticklabels([f"{t:.1f}" for t in new_ticks])
                ax.set_xlabel("Time (ps)")

            # Add a title to the barcode plot
            ax.set_title(f"Interaction barcode for {ligand_selector_string.replace('resname ','')}")
            # Save barcode to PNG
            ax.figure.savefig(f"{self.basename}_{ligand_selector_string.replace('resname ','')}_prolif_barcode.png", dpi=300, bbox_inches="tight")
            #make prolif df
            interaction_df = fp.to_dataframe()

            # Insert Time (ps) as the first column in the DataFrame
            frame_times = [ts.time for ts in u.trajectory[start_frame:stop_frame:sparsity]]  # returns a list of times (usually in picoseconds)
            interaction_df.insert(0, "Time (ps)", frame_times)
            interaction_df.to_csv(f"{self.basename}_{ligand_selector_string.replace('resname ','')}_prolif_df.csv", index=False)

            # Save ligand network plots at different occurrence thresholds
            for threshold in [0.10, 0.30, 0.50, 0.90]:
                lignetwork = fp.plot_lignetwork(ligand_mol, threshold=threshold)
                # fp.plot_lignetwork returns an IPython.display.HTML object, but the HTML content is in lignetwork.data
                with open(f"{self.basename}_{ligand_selector_string.replace('resname ','')}_lignetwork_{threshold}.html", "w") as f:
                    f.write(lignetwork.data)

            # Tanimoto similarity matrix
            bitvectors = fp.to_bitvectors()
            similarity_matrix = []
            for bv in bitvectors:
                similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(bv, bitvectors))
            similarity_matrix = pd.DataFrame(similarity_matrix, index=interaction_df.index, columns=interaction_df.index)
            plot.heatmap(similarity_matrix, x_var="Time (ps)", y_var="Time (ps)", heat_var="Tanimoto similarity", titel="Binding pose similarity", plot_type=f"{ligand_selector_string.replace("resname ", "")}_prolif_2d",  basename=self.basename, 
                               reporting_time =self.reporting_time, sparsity = 1, start_frame = 0)

            interaction_df_dict[ligand_selector_string] = interaction_df
        
        print(f"{round(time.time() - start_time,2)}s used for interaction fingerprints")

        return list(interaction_df_dict.keys()), interaction_df_dict
    
    def save_df_to_csv(self):
        """
        saves time_df and residue_df to csv

        Parameters:
            nothing

        Returns:
            nothing
        """

        self.time_df.to_csv(f"{self.basename}_time.csv", index=False)
        self.residue_df.to_csv(f"{self.basename}_residue.csv", index=False)


if __name__ == "__main__":

    #TODO create a wrapper function of this that presents the user with a equillibrated simulation
    #right away

    #Start the command line parser
    parser = argparse.ArgumentParser(
                        prog='POS MD script',
                        description=(
                        "This script sets up a OPENMM  simulation of a protein (amber ff14)," 
                        "any small molecules/cofactors (Sage 2.2.1) as well as water/ions using a 12-6 model (amber ff14/tip3pfb).\n"),
                        epilog="Use with care and acknowledge Erik Sundén and the Per-Olof Syrén group at KTH Sweden")

    parser.add_argument("pdb", type = str, help = "PDB structure of the structure you want to simulate. \nWARNING PDB may not contain any ligands. These must be provided from sdf files") 
    parser.add_argument("-l", "--lig", type = str, action="append", default = [], help = ("optional parameter, SDF file containing all nonstandard ligands and cofactors." 
                                                                        "Convenientlly produced by drawing in chemdraw and exporting as SDF then docking with added hydrogens."
                                                                        "This is easilly done by checking ChimeraX dockpreps charge assignment when running dockprep."
                                                                        "WARNING charges MUST be assigned in the sdf file, use -cc True to autoassign based on pH"
                                                                        "or whatcat/md/molecule_inspector.ipynb which both converts files and visuallizes result")) 
    parser.add_argument("--restart", type = str, default= "False", choices=["true", "True", "false", "False"], help="Restarts the simulation from restart xml files if set to True \nRequieres that pdb is set to pdbname_final.pdb", required=False)
    parser.add_argument("--platform", type = str, default= "CUDA", choices=["CUDA", "OPENCL", "CPU"], help="Sets the simulation platform, default = CUDA", required=False)

    parser.add_argument("--pdbfixer", type = int, default = 2, help = ("0, 1, 2 depending on if your structure shall be PDBfixed." 
                                                                        "default = 2 removes and readds hydrogens as well as tries to find missing atoms"
                                                                        "good if you have a SEQRES and unresolved loops as well as unhandled disulfide bonds."
                                                                        "=1 fixes loops and so on but retains hydrogens in structure. Good if manual protonation was done"
                                                                        "=0 does not fix your pdb, make sure it is good" ))
    parser.add_argument("-cc", "--charge_correct", type = str, default= "False", choices=["true", "True", "false", "False"], help="Whether to charge correct ligands or not. also converts files to sdf if ligand not sdf")
    parser.add_argument("--solvate", type = int, default= 2, choices=[0,1,2], help="IRegulates solvation. \ndefault = 2 - remove all water and add a solvent box \n 1 = add solvent box \n do not alter solvent", required=False)
    parser.add_argument("-ph", "--ph", type = float, default= 7.4, help="Sets pH for the simulation using PDBfixer and if using -cc openbabel")

    parser.add_argument("-t", "--timeprod", type = float, default= 1, help="Production simulation time in ns. Accepts floats and ints")
    parser.add_argument("-rt", "--report_time", type = float, default= 10, help="Reporting frequency in ps")
    parser.add_argument("-eqt", "--equillibration_time", type = float, default= 50, help="Equillibration time in ps, do not set lower than 50 ps. \nUsed for both NPT and NVT equillibration")
    parser.add_argument("-dt", "--timestep", type = int, default= 4, choices=[1,2,3,4,5], help="Simulation timestep in fs. Accepts ints")
    
    parser.add_argument("--resname", type = str, action="append", default= [], help="Residue names in PDB for which you want further analysis, eg ligand.\n"
                                                                                "several --resnames can be used at once \n if not specified all ligands added with --lig will get analyzed", required=False)
    parser.add_argument("--dist", type = str, action="append", default= [], help="""a pair of atom numbers eg "resid 131 and name OG1, resname UNK and name N1x" for which you want 
    a distance plot eg for monitoring near-attack conformations. specify using MDAnalysis/VMD natural language queries""", required=False)
    
    parser.add_argument("--debug", type = bool, default= False, help="debug mode, prints more information while running", required=False)

    # Parse arguments
    args = parser.parse_args()

    #print command line
    print(f"Parsed arguments: {vars(args)}\n")
    print(parser.description)
    print(parser.epilog + "\n")

    whatcat_md = Whatcat_md_runner(args.pdb, args.lig, utils.str_to_bool(args.restart), args.platform, args.pdbfixer, 
                                   utils.str_to_bool(args.charge_correct), args.solvate, args.ph, args.timeprod, 
                                   args.timestep, args.report_time, args.equillibration_time, args.debug)

    if whatcat_md.restart is not True:
        whatcat_md.fix_pdb()
        whatcat_md.create_openmm_system()
        whatcat_md.create_openmm_simulation()
        whatcat_md.equillibrate_simulation()
    elif whatcat_md.restart is True:
        whatcat_md.restart_simulation_from_file()

    simulation = whatcat_md.run_prod_simulation()

    #set default for analysis
    analysis_distances = args.dist
    analysis_resnames = utils.prepend_list(args.resname, "resname ")

    #if data is availible set that
    if len(args.resname) == 0:
        analysis_resnames = whatcat_md.analysis_resnames

    whatcat_analysis = whatcat_md.create_analysis()
    whatcat_analysis.read_md_log()
    whatcat_analysis.equillibration_check()

    whatcat_analysis.center_align_traj() #this one produces a erroneous DCD file. I suspect the box dimensions are scuffed
    #Using mdconvert to convert the DCD file to a DCD file restores functionallity

    whatcat_analysis.calc_rgyr()
    whatcat_analysis.calc_1d_rmsd(analysis_resnames + ["backbone"])
    whatcat_analysis.calc_ca_rmsf()

    whatcat_analysis.calc_pairwise_distances(analysis_distances)

    whatcat_analysis.write_sparse_traj()
    whatcat_analysis.calc_2d_rmsd(analysis_resnames + ["backbone"])
    whatcat_analysis.remove_sparse_traj()
    whatcat_analysis.run_prolif(analysis_resnames = analysis_resnames)

    whatcat_analysis.save_df_to_csv()

    print(whatcat_analysis.time_df.head())
    print(whatcat_analysis.residue_df.head())


