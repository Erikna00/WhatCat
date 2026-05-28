from openmm.app import *
from openmm import *
import openmm.unit as unit

from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer
import parmed as pmd

import argparse
import numpy as np
import scipy
import pandas as pd
import MDAnalysis as mda
import mdaencore
import mdtraj as mdtraj

import prolif
from rdkit import DataStructs
import matplotlib.pyplot as plt #for plotting broken barh only, move when utility is generalized


from whatcat.md.utils import utils, analysis, plot, efield

import sys
import os
import math
import multiprocessing as mp
import time
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
                 constrain_h = True, simulation_time_ns=None, timestep=4, 
                 reporting_time=10, equillibration_time=50, plot_format="png",
                 pbc=True):
        """

        pdb_file - path to pdbfile, must not contain any small molecules
        ligand - list of paths to ligand file, must be sdf if charge_correct = False, charge_correct converts each ligand with openbabel
        restart - bool for if to restart from _restart.xml files and _final.pdb printed by a previous run. Also appends to existing reporter path
                If restart is set most other parameters will go unused as the class will jump straight to simulation creation
        platform : str
            The platform on which to run the simulation. Legal choices are (case insensitive):
            "CUDA", "HIP", "OpenCL", "CPU"
            If you optionally want to specify a gpu device index eg for a multidie GPU like a AMD MI250x
            append a comma separated string to the end with relevant device indexes e.g. "CUDA_0" or "CUDA_1,2"

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
        plot_format: str
            What matplotlib image format do you want plots to be saved as
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
        self.pbc = pbc
        self.ph = ph

        #ff files and information
        self.small_molecule_ff = 'openff-2.2.1.offxml'
        self.biomolecule_ff = ['amber14-all.xml', 'amber14/tip3pfb.xml']

        self.constrain_h = constrain_h

        self.simulation_time_ns = simulation_time_ns
        self.timestep = timestep #fs
        self.reporting_time = reporting_time #ps
        self.equillibration_time = equillibration_time #ps
        
        self.analysis_resnames = []
        self.plot_format = plot_format

        self.meta = False

        self.pdb_name = os.path.splitext(pdb_file)[0]

        self.temperature = 300 * unit.kelvin
        self.pressure = 1 * unit.bar

        try:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        except:
            self.script_dir = os.getcwd()


        #Restart overrides settings to allow loading of pdb_final
        if restart == True:
            self.pdb_name = self.pdb_name.replace("_final", "")

    def fix_pdb(self):
        """
        Converts a PDBfile to a openmm PDB object on all settings.
        Uses self.pdbfixer as the default setting but this can be overridden via the passed args
        if so self.pdbfixer is revised to the new value

        Saves a openmm PDBfile object as a class variable
        """

        if self.pdb_fixer == 1 or self.pdb_fixer == 2:

            if self.pdb_fixer == 2:
                #we remove and then re-add hydrogens to prevent shenanigans related to disulfide bonds
                utils.remove_hydrogens(self.pdb_file, f"{self.pdb_name}_fixed.pdb")

                #Run PDBfixer
                fixer = PDBFixer(filename=f"{self.pdb_name}_fixed.pdb")
            
            elif self.pdb_fixer == 1:
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

            PDBFile.writeFile(fixer.topology, fixer.positions, open(f"{self.pdb_name}_fixed.pdb", 'w'))

            
            pdb = PDBFile(f"{self.pdb_name}_fixed.pdb")

        elif self.pdb_fixer == 0:
            #if not fixing PDB
            pdb = PDBFile(self.pdb_file)

        else:
            raise ValueError("illegal option choosen for pdbfixer, valid options are 0, 1, 2")
        
        self.pdb = pdb
    
    def create_openmm_system(self, variants = None):
        """
        Creates a openmm_system and modeller object saved in the class object

        Parameters:
            variants: dict
                A dictionary mapping residue indices to variant names for use in the modeller.addHydrogens() function. 
                This is used to specify special residue types such as disulfide bonded cysteines (CYX) or specific histidine protonation states (HIE, HID, HIP).
                Zero indexing is used for residue indices here.
        """

        #forcefield kwargs
        forcefield_kwargs = {'hydrogenMass' : 1.5 * unit.amu,'constraints': HBonds, 'rigidWater': True, 'removeCMMotion': True}
        
        #No HMR and no constraints if not constraining hydrogens
        if self.constrain_h == False:
            forcefield_kwargs['constraints'] = "None"
            forcefield_kwargs['rigidWater'] = False

        #If using PBC we dont need to change anything
        if self.pbc:
            nonperiodic_forcefield_kwargs = None
        #No PBC
        else: 
            nonperiodic_forcefield_kwargs = {'nonbondedMethod': NoCutoff}

        #if simulating with ligand
        if len(self.ligand_files) > 0:
            cache_file = f"{self.script_dir}/ligands.json"
            lig_namer = utils.Ligand_namer()
            ligand_mol = []
            lig_resnames = []

            #start a cache finder object
            cache_finder = utils.FF_cache_reader(cache_file=cache_file)

            for lig in self.ligand_files:
                
                if self.charge_correct == True:
                    #TODO interface openeye as the primary pKa engine using either pkatyper or openff-toolkit.enumerate_protomers
                    #https://docs.eyesopen.com/toolkits/python/quacpactk/pkatypertheory.html
                    lig = utils.prepare_ligand_md(lig, self.ph)

                #read ligand file into a openFF molecule object
                ligand = Molecule.from_file(lig)

                #convert to openff smiles convention and check against cache
                if not cache_finder.check_smiles_in_cache(ligand.to_smiles(), forcefield=self.small_molecule_ff):
                    #This try-except was added due to GGPP -3 crashing during parametrization
                    #TODO this is bad but is needed for highlly charged ligands see https://github.com/openforcefield/openff-toolkit/issues/1741 https://github.com/openforcefield/openff-toolkit/issues/1911
                    #TODO Maybe wait for resolution of pull requests here? Issue likelly is #1911 with SCF not converging for GGPP -3?
                    #Alt use Psi4 OpenFF Recharge to interface Psi4 or something?
                    #Alt get openeye https://docs.openforcefield.org/projects/toolkit/en/latest/api/generated/openff.toolkit.topology.Molecule.html#openff.toolkit.topology.Molecule.assign_partial_charges
                    try:
                        print("assigning charges")
                        ligand.assign_partial_charges("am1bcc")
                        print("charges assigned with Am1")
                    except:
                        print("\n WARNING\nam1bcc failed, falling back to Gasteiger charges\n")
                        ligand.assign_partial_charges("gasteiger")
                        print("charges assigned with Gasteiger")

                lig_name = lig_namer.name_ligand(os.path.basename(lig))

                #Name the openff molecule
                ligand.name = lig_name

                #set name for atoms in residue to get desired behaviour from OpenFFtoolkit
                for atom in ligand.atoms:
                    atom.metadata['residue_name'] = ligand.name
                
                #keep track of what ligands we are handling
                lig_resnames.append(ligand.name) 

                #add to list which will be added to topology
                ligand_mol.append(ligand)


            # Specify the forcefield
            # Initialize a SystemGenerator using the Sage.2.1 for the ligand and tip3p for the water.
            system_generator = SystemGenerator(
                forcefields=self.biomolecule_ff,
                small_molecule_forcefield=self.small_molecule_ff,
                molecules=ligand_mol, nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
                forcefield_kwargs=forcefield_kwargs, cache=cache_file)
            
            #start a modeller
            modeller = Modeller(self.pdb.topology, self.pdb.positions)

            if self.pdb_fixer != 0:
                #add hydrogens according to user specifications
                if variants is None:
                    variant_lst = None
                else:
                    #default is None, aka default workflow
                    variant_lst = [None for i in modeller.topology.residues()]
                    for res_idx, variant in variants.items():
                        variant_lst[res_idx] = variant

                modeller.addHydrogens(system_generator.forcefield, pH=self.ph, variants=variant_lst)

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
                forcefields=self.biomolecule_ff,
                small_molecule_forcefield=self.small_molecule_ff,
                nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
                forcefield_kwargs=forcefield_kwargs)
            
            #start a modeller
            modeller = Modeller(self.pdb.topology, self.pdb.positions)

            if self.pdb_fixer != 0:
                #add hydrogens according to user specifications
                if variants is None:
                    variant_lst = None
                else:
                    #default is None, aka default workflow
                    variant_lst = [None for i in modeller.topology.residues()]
                    for res_idx, variant in variants.items():
                        variant_lst[res_idx] = variant
                
                modeller.addHydrogens(system_generator.forcefield, pH=self.ph, variants=variant_lst)


        if self.solvate == 2:
            #remove all water
            modeller.deleteWater()
        if self.solvate > 0:
            #add solvent box
            modeller.addSolvent(system_generator.forcefield, padding=1.0*unit.nanometer)
        
            #Purge box vectors added when adding solvent to avoid simulation turning periodic again
            if self.pbc == False:
                modeller.topology.setPeriodicBoxVectors(None)

        # Create the system using the SystemGenerator
        system = system_generator.create_system(modeller.topology)
        
        #Check pbc was correctlly set
        if system.usesPeriodicBoundaryConditions() != self.pbc:
            raise ValueError(f"OpenMM thinks there is a PBC (system.pbc = {system.usesPeriodicBoundaryConditions()}) but you requested there not to be one. Post a bug report with complete input."
                             "This is likelly due to --solvation not 0")

        self.system = system
        self.modeller = modeller
    
    def create_openmm_simulation(self):
        """
        Creates and returns a openmm simulation object ready for use with NVT
        """
        #set precision and platform
        platform_name, device_idx = utils.platform_device_sanitizer(self.platform)

        if platform_name == "OpenCL" or platform_name == "CUDA" or platform_name == "HIP":
            # mixed precision improves energy conservation resulting in larger stable timesteps. decreases speed ca 10%
            properties = {"Precision":"mixed", "DeviceIndex":device_idx} 
            platform = Platform.getPlatformByName(platform_name)
        elif platform_name == "CPU":
            properties = {}
            platform = Platform.getPlatformByName(platform_name)

        #set up simulation
        self.integrator = LangevinMiddleIntegrator(self.temperature, 1/unit.picosecond, self.timestep * unit.femtoseconds)
        simulation = Simulation(self.modeller.topology, self.system, self.integrator, platform = platform, platformProperties=properties)
        simulation.context.setPositions(self.modeller.positions)

        self.simulation = simulation
    
    def restart_simulation_from_file(self): 
        """
        Reads the xml restart files and restarts a simulation object from the same.
        """

        #Remove file ending if present due to bad user input
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
        self.timestep = round(simulation.integrator.getStepSize().value_in_unit(unit.femtosecond), 2)
        print(f"\nSimulation restarted with stepsize of {self.timestep} fs")

        self.simulation = simulation
        
        #Lastlly, Try to read back resnames for small molecules for analysis
        #save all ligands for analysis
        if len(self.ligand_files) > 0:
            lig_resnames = []
            lig_namer = utils.Ligand_namer()

            for ligands in self.ligand_files:
                ligand_name = lig_namer.name_ligand(os.path.basename(ligands))
                lig_resnames.append(ligand_name)

            for ligand_name in lig_resnames:
                self.analysis_resnames.append(f"resname {ligand_name}")
    
    def equillibrate_simulation(self, minimize=True):
        """
        Takes a openmm simulation object and equillibrates it for the provided time converting it
        to a NPT simulation in the process.
        Prints equillibration step log to md_log_equil.txt
        """

        equillibration_steps = int(self.equillibration_time / (self.timestep * 10**-3))

        if minimize == True:
            print("Minimizing energy")
            self.simulation.minimizeEnergy()

        #add equillibration reporter
        reporting_frequency = int(self.reporting_time / (self.timestep * 10**-3))
        self.simulation.reporters.append(StateDataReporter(f"{self.pdb_name}_md_log_equil.txt", reporting_frequency, step=True,
        potentialEnergy=True, temperature=True, volume=True, speed=True, remainingTime=True, totalSteps=equillibration_steps * 2, append = self.restart))
        self.simulation.reporters.append(DCDReporter(f"{self.pdb_name}_trajectory_equil.dcd", reporting_frequency, append = self.restart))

        print("Running NVT equillibration")
        self.simulation.step(equillibration_steps)

        #If using PBC we want to be move to NPT ensamble
        if self.pbc:
            self.system.addForce(MonteCarloBarostat(self.pressure, self.temperature))
            self.simulation.context.reinitialize(preserveState=True) #needed to add in the barostat

            print("Running NPT equillibration")
            self.simulation.step(equillibration_steps)
        
        #If not using PBC we remain in NVT
        else:
            print("Skipping NPT equillibration since PBC is disabled")

        #save pdb
        state = self.simulation.context.getState(getPositions=True)
        with open(self.pdb_name + "_equillibrated.pdb", "w") as file:
            PDBFile.writeFile(self.simulation.topology, state.getPositions(), file)

        #reset simulation time to 0 for decent analysis
        self.simulation.currentStep = 0  # Reset step counter
        self.simulation.context.setTime(0 * unit.picoseconds)  # Reset simulation time to 0 ps

        #remove equillibration reporter
        self.simulation.reporters.clear()

        self.equillbration_steps = equillibration_steps
    
    def add_metadynamics(self, atom_indices, colvar_parameters, bias_factor = 10, hill_height=1.0, hill_frequency=500, shift_dihedrals=False):
        """
        Adds a metadynamics bias to the system using OpenMM's Metadynamics class.
        Also creates the collective variable (CV) force.
        Sets self.metadynamics containing a OPENMM Metadynamics

        Parameters:
            atom_indices: list
                Atom indices for the CV. For "bond", provide [i, j]. For "angle", provide [i, j, k].
            colvar_parameters: list of lists
                Parameters for each colvar [[min_value, max_value, hill_width] ...]
            bias_factor: float
                Bias factor for well-tempered metadynamics.
            hill_height: float
                Height of the deposited hills (in kJ/mol).
            hill_frequency: int
                How often (in steps) to deposit a hill.

        """

        if not hasattr(self, "simulation"):
            raise RuntimeError("Simulation must be created before adding metadynamics.")

        # Create the CV forces
        # We do it this way since periodic colvars cannot be mixed with non periodic colvars
        periodic_list =[]
        cv_forces = []
        units = []
        bias_variables = []

        #sanitize colvar parameters
        colvar_parameters = utils.colvar_sanitizer(colvar_parameters)

        #save MTD parameters in class
        self.colvar_parameters = colvar_parameters
        self.atom_indices = atom_indices

        for index in range(0, len(atom_indices)):

            if len(atom_indices[index]) == 2:
                # Harmonic bond CV
                cv = CustomBondForce("r")
                cv.addBond(atom_indices[index][0], atom_indices[index][1], [])
                cv_force = CustomCVForce("bond")
                cv_force.addCollectiveVariable("bond", cv)
                units.append(unit.angstrom)
                periodic_list.append(False)
            
            elif len(atom_indices[index]) == 3:
                # Harmonic angle CV
                cv = CustomAngleForce("theta")
                cv.addAngle(atom_indices[index][0], atom_indices[index][1], atom_indices[index][2], [])
                cv_force = CustomCVForce("angle")
                cv_force.addCollectiveVariable("angle", cv)
                units.append(unit.degree)
                periodic_list.append(False)

            elif len(atom_indices[index]) == 4:
                #Dihedral CV
                if shift_dihedrals is False:
                    cv = CustomTorsionForce("theta")
                if shift_dihedrals is True:
                    #shift dihedral to be between 0 and 360 degrees
                    cv = CustomTorsionForce("(theta + pi) - 2*pi*floor((theta + pi)/(2*pi))")
                    cv.addGlobalParameter("pi", math.pi)
                cv.addTorsion(atom_indices[index][0], atom_indices[index][1], atom_indices[index][2], atom_indices[index][3], [])
                cv_force = CustomCVForce("dihedral")
                cv_force.addCollectiveVariable("dihedral", cv)
                units.append(unit.degree)
                periodic_list.append(True)

            else:
                raise ValueError("cv_type must be 'bond', 'angle', or 'dihedral'. Something went wrong in your mtd_cv specification")
            
            #add cv_force to collection
            cv_forces.append(cv_force)

        periodic = all(periodic_list)  # Check if all CVs are periodic
        if not periodic and shift_dihedrals is False and any("dihedral" in cv.getEnergyFunction() for cv in cv_forces):
            print("Running metadynamics with non-periodic CVs. This may lead to unexpected behavior for dihedrals since there will be a discontinuity at 180 and -180 degrees. " \
            "Consider using shift_dihedrals=True to instead have this issue at 0 and 360 degrees.")
        elif periodic and shift_dihedrals is True:
            print("There is no need to shift dihedrals when using periodic CVs, since they are already periodic and thus lack a discontinuity")
        elif periodic:
            print("Running metadynamics with periodic CVs")

        for index in range(0, len(cv_forces)):
            # Wrap cv_forces in a BiasVariable object
            #here we employ a hack to allow for dihedrals to be shifted to 0-360 degrees
            if cv_forces[index].getEnergyFunction() != "dihedral" or shift_dihedrals is False:
                bias_variable = BiasVariable(cv_forces[index], colvar_parameters[index][0] * units[index], 
                                            colvar_parameters[index][1] * units[index], colvar_parameters[index][2] * units[index], periodic=periodic)
                
            elif cv_forces[index].getEnergyFunction() == "dihedral" and shift_dihedrals is True:
                bias_variable = BiasVariable(cv_forces[index], (colvar_parameters[index][0] + 180) * units[index], 
                                            (colvar_parameters[index][1] + 180) * units[index], colvar_parameters[index][2] * units[index], periodic=periodic)
                
            else:
                raise ValueError("Dihedral shifting error, save commandline and files and make a bug report")
            
            bias_variables.append(bias_variable)

        # Create the Metadynamics object
        metadynamics = Metadynamics(
            system=self.simulation.system,
            variables=bias_variables,
            temperature=self.temperature,
            biasFactor=bias_factor,
            height=hill_height * unit.kilojoule_per_mole,
            frequency=hill_frequency,
            saveFrequency= 1 * hill_frequency,
            biasDir= os.getcwd() #take biases from CLI position
        )

        #reinitiallize simulation to add in the metadynamcis
        self.simulation.context.reinitialize(preserveState=True)

        self.meta = True
        self.metadynamics = metadynamics

    def run_metadynamics_simulation(self):
        """
        Runs the production metadynamics simulation for the set amount of time.

        """

        #calculate simulation length
        production_steps = int(self.simulation_time_ns / (self.timestep * 10**-6))
        reporting_frequency = int(self.reporting_time / (self.timestep * 10**-3))
        steps_at_finish = production_steps + self.simulation.context.getStepCount()

        #add reporters
        #print to terminal
        self.simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
                potentialEnergy=True, temperature=True, volume=True, remainingTime=True, totalSteps= steps_at_finish, speed=True))

        #saved to file
        # Check if output files already exist and revise append accordinglly
        self.traj_file = f"{self.pdb_name}_trajectory_metadynamics.dcd"
        self.md_log = f"{self.pdb_name}_md_log_metadynamics.txt"

        #Indicate to the user if files are missing for a restart
        if self.restart and (not os.path.exists(self.md_log) or not os.path.exists(self.traj_file)):
            print("Restart is set to True but md_log_metadynamics.txt or trajectory_metadynamics.dcd does not exist. I hope this was intended")

        append = os.path.exists(self.md_log) and os.path.exists(self.traj_file) and self.restart
        append = append and os.path.getsize(self.traj_file) > 0 # only append if DCD file is non empty, otherwise we get header errors

        self.simulation.reporters.append(StateDataReporter(self.md_log, reporting_frequency, step=True,
            potentialEnergy=True, temperature=True, volume=True, append = append))
        self.simulation.reporters.append(DCDReporter(self.traj_file, reporting_frequency, append = append))

        #since MTD is prone to crashing we run 1 ns at a time
        dump_freq = 0.1 #ns
        steps_per_cycle = int(dump_freq * 1e6 // self.timestep)
        cycles_to_run = production_steps // steps_per_cycle
        remainder_steps = production_steps % steps_per_cycle

        pes_last = None
        pes_rmsd_lst = []
        simulation_dump_lst = []

        #if restarting read already dumped csv file and pes
        if self.restart and os.path.exists(f"{self.pdb_name}_metadynamics_pes.csv") or os.path.exists(f"{self.pdb_name}_metadynamics_pes.npy") and f"{self.pdb_name}_mtd_convergence.csv":
            pes_rmsd_lst, simulation_dump_lst, pes_last = utils.metadynamics_pes_convergence_reader(self.pdb_name)

        print("Running production NPT metadynamics")
        for cycle in range(cycles_to_run):
            self.metadynamics.step(self.simulation, steps_per_cycle) # run dump_freq ns at a time

            print("dumped checkpoint")

            #dump restart files
            self.dump_restart_files()

            pes = self.metadynamics.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)
            pes = pes - np.min(pes) #shift the  free energy so lowest energy is 0 kJ/mol

            if pes_last is not None:
                #calculate rmsd of PES towards last pes to see if it is converged
                pes_rmsd = np.sqrt(np.mean(np.exp2(pes - pes_last))) 
                pes_rmsd_lst.append(pes_rmsd)
                simulation_dump_lst.append(self.simulation.context.getTime().value_in_unit(unit.nanosecond)) 
                plot.line_plotter_2d(simulation_dump_lst, pes_rmsd_lst, "Simulation time (ns)", "PES RMSD (kJ/mol)", 
                                     self.pdb_name, "mtd_convergence", force_zero_start_x=True, 
                                     force_zero_start_y=True, plot_format=self.plot_format)
                
            pes_last = pes
            self.save_mtd_pes(pes)
            plot.metadynamics_plotter(pes, self.atom_indices, self.colvar_parameters, self.pdb_name, plot_format=self.plot_format)

            #save convergence info to csv
            pes_rmsd_df = pd.DataFrame({"Simulation time (ns)": simulation_dump_lst, "PES RMSD (kJ/mol)": pes_rmsd_lst})
            pes_rmsd_df.to_csv(f"{self.pdb_name}_mtd_convergence.csv", index=False)
            
        if remainder_steps > 0 or production_steps == 0:
            self.metadynamics.step(self.simulation, remainder_steps)

            #dump restart files
            self.dump_restart_files()

            pes = self.metadynamics.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)
            pes = pes - np.min(pes) #shift the  free energy so lowest energy is 0 kJ/mol

            if pes_last is not None:
                #calculate rmsd of PES towards last pes to see if it is converged
                pes_rmsd = np.sqrt(np.mean(np.exp2(pes - pes_last))) 
                pes_rmsd_lst.append(pes_rmsd)
                simulation_dump_lst.append(self.simulation.context.getTime().value_in_unit(unit.nanosecond)) 
                plot.line_plotter_2d(simulation_dump_lst, pes_rmsd_lst, "Simulation time (ns)", "PES RMSD (kJ/mol)", self.pdb_name, "mtd_convergence", force_zero_start_x=True, force_zero_start_y=True)

            self.save_mtd_pes(pes)
            plot.metadynamics_plotter(pes, self.atom_indices, self.colvar_parameters, self.pdb_name)

            #save convergence info to csv
            pes_rmsd_df = pd.DataFrame({"Simulation time (ns)": simulation_dump_lst, "PES RMSD (kJ/mol)": pes_rmsd_lst})
            pes_rmsd_df.to_csv(f"{self.pdb_name}_mtd_convergence.csv", index=False) 

        return pes

    def save_mtd_pes(self, pes):
        """
        Saves the potential energy surface (PES) from metadynamics to a file.
        If the PES is 1D or 2D, it saves as a CSV file. (yx indexed for 2D)
        If the PES is 3D, it saves as a NumPy array file. (zyx indexed)

        Parameters:
            pes: np.ndarray
                The potential energy surface data.
            colvar_parameters: list of lists
                Parameters for each colvar [[min_value, max_value, hill_width] ...]
        """
        colvar_parameters = self.colvar_parameters

        if len(pes.shape) == 1:
            # 1D case
            colvar_range = np.linspace(colvar_parameters[0][0], colvar_parameters[0][1], pes.shape[0])
            pes_df = pd.DataFrame({"colvar": colvar_range, "free_energy": pes})
            pes_df.to_csv(f"{self.pdb_name}_metadynamics_pes.csv", index=False)

        elif len(pes.shape) == 2:
            # 2D case
            x_range = np.linspace(colvar_parameters[0][0], colvar_parameters[0][1], pes.shape[0])
            y_range = np.linspace(colvar_parameters[1][0], colvar_parameters[1][1], pes.shape[1])
            pes_df = pd.DataFrame(pes, index=x_range, columns=y_range)
            pes_df.index.name = "colvar_1"
            pes_df.columns.name = "colvar_2"
            pes_df.to_csv(f"{self.pdb_name}_metadynamics_pes.csv", index=True)

        #If data is 3D we save as a  npy array file
        elif len(pes.shape) == 3:
            np.save(f"{self.pdb_name}_metadynamics_pes.npy", pes)

    def run_prod_simulation(self):
        """
        Runs the production NPT simulation for the set amount of time.
        """

        #calculate simulation length
        production_steps = int(self.simulation_time_ns / (self.timestep * 10**-6))
        reporting_frequency = int(self.reporting_time / (self.timestep * 10**-3))
        steps_at_finish = production_steps + self.simulation.context.getStepCount()

        #add reporters
        #print to terminal
        self.md_log = f"{self.pdb_name}_md_log.txt"
        self.traj_file = f"{self.pdb_name}_trajectory.dcd"

        
        #Indicate to the user if files are missing for a restart
        if self.restart and (not os.path.exists(self.md_log) or not os.path.exists(self.traj_file)):
            print("Restart is set to True but md_log_metadynamics.txt or trajectory_metadynamics.dcd does not exist. I hope this was intended")
        

        self.simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
                potentialEnergy=True, temperature=True, volume=True, remainingTime=True, totalSteps= steps_at_finish, speed=True))

        #saved to file
        self.simulation.reporters.append(StateDataReporter(self.md_log, reporting_frequency, step=True,
                potentialEnergy=True, temperature=True, volume=True, append = self.restart))
        self.simulation.reporters.append(DCDReporter(self.traj_file, reporting_frequency, append = self.restart))

        if self.pbc == True:
            print("Running production NPT")
        else:
            print("Running production NVT since PBC is disabled")
        self.simulation.step(production_steps)

        #dump restart files
        self.dump_restart_files()

    
    def dump_restart_files(self):
        """
        Dumps the simulation state, system and integrator to XML files.
        Also saves the final PDB file of the simulation.
        """

        #save pdb
        state = self.simulation.context.getState(getPositions=True)
        with open(f"{self.pdb_name}_final.pdb", "w") as file:
            PDBFile.writeFile(self.simulation.topology, state.getPositions(), file)
        
        #save checkpoints of state, system and integrator
        checkpoint_filebase = f"{self.pdb_name}_restart"
        self.simulation.saveState(f"{checkpoint_filebase}_state.xml")

        with open(f"{checkpoint_filebase}_system.xml", 'w') as file:
            file.write(XmlSerializer.serialize(self.system))
        with open(f"{checkpoint_filebase}_integrator.xml", 'w') as file:
            file.write(XmlSerializer.serialize(self.integrator))

    def dump_amber_files(self):
        """ 
        Dumps the prmtop and inpcrd files for AMBER simulations.
        This is useful for running AMBER simulations from the same system.
        """
        # Use ParmEd to convert OpenMM system/modeller to AMBER files
        state = self.simulation.context.getState(getPositions=True)
        parm = pmd.openmm.load_topology(self.simulation.topology, self.system, xyz=state.getPositions())
        prmtop_file = f"{self.pdb_name}.prmtop"
        inpcrd_file = f"{self.pdb_name}.inpcrd"
        parm.save(prmtop_file, overwrite=True)
        parm.save(inpcrd_file, format="rst7", overwrite=True)
        print(f"Saved AMBER files: {prmtop_file}, {inpcrd_file}")


    def create_analysis(self, plot = True, max_sparse_frames=500, ncores = 0):
        """
        Creates a Whatcat_md_analysis object based on the simulation and returns it
        Exports pdb_name, simulation.topology, trajectory filename, reporting time and simulation time to analyzer

        Parameters
            plot : bool
                Do we want graphical plots or not from the analysis
            max_sparse_frames : int
                How many frames can the sparse trajectory contain at most.
            ncores:int
                How many cores should the analysis use?
                0= use all availible
        """

        md_analysis = Whatcat_md_analysis(self.pdb_name, self.simulation.topology, self.traj_file, self.md_log,
                                          self.reporting_time, plot = plot, max_sparse_frames=max_sparse_frames, ncores=ncores, 
                                          platform=self.platform, plot_format=self.plot_format)

        return md_analysis


class Whatcat_md_analysis:
    
    def __init__(self, basename, topology, traj_file, md_log, reporting_time, 
                 align = True, plot = True, 
                 max_sparse_frames = 500, ncores = 0, platform="CUDA",
                 plot_format = "png"):
        """
        This class analyzes MD simulations by wrapping MDAnalysis in a parallelized executor using
        divide and conquer methodologies when the MDAnalysis function does not have native parallelization
        Prints out a pdb file of the topology if self.basename_final.pdb does not exist

        Parameters
            basename : str
                The base name for the simulation files, typically the name of the PDB file without the extension
                If generated using whatcat_md_runner it is the name of the initial pdb file without .pdb
            topology : str or OPENMM topology compatible with MDAnalysis 
                A PDB file or OPENMM topology. Needs to contain bonding information for center_align_traj to work
                This is fulfilled by passing simulation.topology from openmm
            traj_file : str
                A trajectory file such as .dcd
            md_log : str
                A log file from OPENMM state data reporter
            reporting_time : int
                The reporting time in ps used in the simulation
            metadynamics : Bool
                If the simulation was run with metadynamics or not
            align : Bool 
                if you want trajectory aligned to first frame
            plot : Bool
                if you want plots or only save to df
            Max_sparse_frames : int
                Maximum number of frames in sparse trajectory used for N^2 scaling analysis like 2D RMSD
            ncores : int
                Number of cores to use for parallelization. Default 0 uses all available cores
            platform: str
                The OPENMM platform which you want efield decomposition to run on
            plot_format: str
                What matplotlib image format do you want plots to be saved as
        
        Returns
            A whatcat_md_analysis object
        """
        self.traj_file = traj_file
        self.topology = topology
        self.u = mda.Universe(topology, traj_file, dt=reporting_time)

        self.basename = basename
        self.avg_pdb_name=f"{self.basename}_avg_structure.pdb"
        self.avg_pdb_generated = False
        self.align = align
        self.plot = plot
        self.plot_format = plot_format

        #Make sure self.pdb_file is set to a valid PDB file or make one
        if os.path.exists(f"{self.basename}_final.pdb"):
            self.pdb_file = f"{self.basename}_final.pdb"

        elif isinstance(self.topology, str) and ".pdb" in self.topology:
            self.pdb_file = self.topology

        elif isinstance(self.topology, Topology):
            self.u.trajectory[-1]
            self.u.atoms.write(f"{self.basename}_final.pdb")
            self.u.trajectory[0]

        else:
            raise ValueError("topology must be a PDB file or an OPENMM Topology object")

        self.reporting_time = reporting_time
        self.md_log = md_log

        #Max frames in sparse traj, regulates N^2 scaling analysis like 2D RMSD
        self.max_sparse_frames = max_sparse_frames
        
        #Values belonging to sparse traj. Not initiallized until sparse traj is written
        self.sparse_traj = None
        self.sparse_traj_start_frame = None
        self.sparse_traj_end_frame = None #stored as the actual end frame so dont forget to +1 when slicing.
        self.sparsity = None
        self.sparse_time_ps = None

        self.time_df = pd.DataFrame()
        
        #start residue df with topology attributes
        protein = self.u.select_atoms("protein")
        #+1 to get to pdb resids
        self.residue_df = pd.DataFrame({"resid_idx": protein.residues.ix_array +1,
                                   "resname": protein.residues.resnames,
                                   "chainid": [residue.segment.segid for residue in protein.residues]
                                   })
        if ncores > 0:
            self.n_jobs = ncores
        else:
            #get how much we can parallelize
            self.n_jobs = mp.cpu_count()

        self.platform = platform

    def read_md_log(self):
        """
        Reads the log file from whatcat_md_runner into self.time_df

        Parameters:
            None

        Returns:
            None but adds information to _time df
        """

        u = self.u

        #load state data reporter information from memory
        column_names = ["Step", "Potential Energy (kJ/mole)", "Temperature (K)", "Box Volume (nm^3)"]
        
        #header=None prevents using the first row as headers avoiding "#"steps" as a name
        #nrows = None allows us to read the whole csv not just the first 100 rows
        self.time_df = pd.read_csv(self.md_log, names=column_names, comment="#", header=None, nrows=None)  

        # Extract Trajectory Time
        times = np.array([ts.time for ts in u.trajectory])  # Extract time (in ps)
        self.time_df.insert(0, "Time (ps)", times)

        if self.plot:
            #plot reporter info from OPENMM
            plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df["Potential Energy (kJ/mole)"], "Time (ps)", "Potential Energy (kJ/mole)", 
                                 self.basename, "energy", histogram = True, plot_format=self.plot_format)
            plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df["Temperature (K)"], "Time (ps)", "Temperature (K)", 
                                 self.basename, "temperature", histogram = True, plot_format=self.plot_format)
            plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df["Box Volume (nm^3)"], "Time (ps)", "Box Volume (nm^3)", 
                                 self.basename, "volume", histogram = True, plot_format=self.plot_format)
        
        #for sanity check we calculate timestep from md_log and reporting time and compare to what is reasonable for MD
        dt = self.time_df["Time (ps)"].iloc[1] - self.time_df["Time (ps)"].iloc[0]
        dstep = self.time_df["Step"].iloc[1] - self.time_df["Step"].iloc[0]
        timestep = dt*1000 / dstep
        if timestep > 6 or timestep < 0.5:
            print(f"\nWARNING: After calculating simulation timestep from md_log and given reporting time, the timestep is {timestep} fs. This is unusual and might indicate an error in the log file or more likelly that you have misreported the reporting time.\n")

    def center_align_traj(self):
        """
        Centers the protein in the periodic box.
        Optionally aligns the protein to the first frame to remove tumbling
        Aligning is necessary for some of the other analyses in this class to work
        Overwrites self.pdb_file and self.traj_file with centered and aligned versions
        self.pdb_file is written as the final frame of the trajectory

        Parameters

        Returns
            Nothing and does not add to df
        """
        
        print("Centering and aligning protein in PBC" if self.align else "Centering protein in PBC")

        start_time = time.time()

        #center proteins, This also resets the time of the first snapshot to 0 ps thus removing equillibration time
        #this unintended behavior is no longer utilized since we reset time after equillibration
        centered_traj_name = self.traj_file
        final_pdb = self.pdb_file

        mda_traj = utils.parallel_center_trajectory(self.topology, self.traj_file, align=self.align, n_jobs=self.n_jobs, output_filename = centered_traj_name) 

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

        md_traj = mdtraj.load(centered_traj_name, top=final_pdb) #MDtraj cannot ingest a OPENMM system as topology   
        md_traj.save_dcd(centered_traj_name)

        print(f"{round(time.time() - start_time,2)}s used for centering")

    def load_avg_structure(self):
        """
        If avg_pdb_name exists on disk, load it as a MDAnalysis Universe and return it.
        If it does not exist, calculate the average structure from the provided Universe u,
        save it to avg_pdb_name, load it as a Universe and return it.

        Structure is average for protein CA atoms only, excluding ACE and NME capping groups.
        Changing this would require chaning RMSF and distance calculation as well.

        Parameters:
            avg_pdb_name : str
                Path to the average structure PDB file.
            u : MDAnalysis.core.universe.Universe
                The MDAnalysis Universe from which to calculate the average structure if needed.
                Needs to include both topology and trajectory.
        Returns:
            MDAnalysis.core.universe.Universe
                The Universe containing the average structure only with no trajectory.

        """

        if self.avg_pdb_generated == True:
            avg_u = mda.Universe(self.avg_pdb_name) #load the average structure as universe

        elif self.avg_pdb_generated == False:
            print("computing average structure")
            start_time = time.time()

            #DONT FUCK WITH THIS SELECTION WITHOUT LOOKING AT DOWNSTREAM EFFECTS
            selection = "protein and name CA and not resname ACE NME"
            u = self.u

            #Calculate the average structure if it does not exist
            avg_u = utils.parallel_average_structure(self.pdb_file, self.traj_file, 
                                                     selection, n_jobs=self.n_jobs, output_filename=f"{self.basename}_avg_structure.pdb")

            # Set unit cell dimensions from the first frame before writing
            #avg_struct.dimensions = u.trajectory[0].dimensions

            self.avg_pdb_generated = True
            print(f"{round(time.time() - start_time,2)}s used for average structure")
        
        return avg_u

    def calc_geometric_params(self, analysis_selections):
        """
        Calculates geometric parameters between specified atom selections for a trajectory.
        Each element of `atom_pairs` should be a string with two selection
        queries separated by a comma.
        
        Parameters
            analysis_selections : np.ndarray or list of str
                Array/list of strings specifying atom selections (MDAnalysis selection language) in the format:
                "selection1, selection2"
                eg "resid 131 and name OG1, resname UNK and name N1x" for a distance.
                2 atoms/selections = distance, 3 atoms/selections = angle, 4 atoms/selections = dihedral
        
        Returns
            np.ndarray
                2D NumPy array of shape (num_frames, num_pairs) where each row contains the
                computed distances for the specified pairs in that frame.
            also adds columns to self.time_df
        """

        #if empty input
        if len(analysis_selections) == 0:
            return None
        
        #compute distances
        print("computing geometric parameters")
        start_time = time.time()

        # Create the MDAnalysis Universe.
        u = self.u

        #unpack the list of comma separated selection strings to a list of list of selectors
        atom_selector_lists = [utils.css_to_list(selection_str) for selection_str in analysis_selections]
        
        # Set up AnalysisFromFunction.
        # Here, we pass u.trajectory as the trajectory to iterate over and u.atoms as the AtomGroup
        # to be updated on each frame. The 'pair_selections' tuple is passed as an argument to our function.
        work = mda.analysis.base.AnalysisFromFunction(analysis.compute_geometric_params_frame, u.trajectory, u.atoms, atom_selector_lists)
        work.run(backend = "multiprocessing", n_workers = self.n_jobs)

        self.time_df = pd.concat([self.time_df, pd.DataFrame(work.results.timeseries, columns = analysis_selections)], axis = 1)

        #figure out what is a distance
        isdist = [len(sel) == 2 for sel in atom_selector_lists]

        if self.plot:
            # Filter selection strings to include only those that are distances
            distance_selections = [sel_str for sel_str, is_dist in zip(analysis_selections, isdist) if is_dist]
            if any(distance_selections):
                plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df[distance_selections], "Time (ps)", "Distance (Å)", 
                                     self.basename, "distances", force_zero_start_x=True, histogram = True, plot_format=self.plot_format)

            # Filter selection strings to include only those that are distances
            angle_selections = [sel_str for sel_str, is_dist in zip(analysis_selections, isdist) if not is_dist]
            if any(angle_selections):
                plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df[angle_selections], "Time (ps)", "Angle (degrees)", 
                                     self.basename, "angles", force_zero_start_x=True, histogram = True, plot_format=self.plot_format)


        print(f"{round(time.time() - start_time,2)}s used for geometric parameters")

        return work.results.timeseries
    
    def calc_ca_rmsf(self, colored_pdb_ending = None):
        """
        Compute RMSF with on-the-fly trajectory alignment using MDAnalysis transformations, 
        ensuring low memory usage and parallelization by processing frames in parallel.

        Parameters
            colored_pdb_ending : str
                During this analysis a PDB colored according to rmsf is written to basename + colored_pdb_ending
                Default reads in self.pdb_file and overwrites it with the new information
            Selection : str
                The selection for which RMSF is calculated. No matter the selection, the alignment and averaging of structures 
                is done using "protein and name CA"

        Returns:
            np.ndarray: Computed RMSF values per residue.
        """
        #compute RMSF
        print("computing protein Ca RMSF")
        start_time = time.time()

        # Load Universe
        u = self.u
        
        # Compute the average structure (for alignment reference)
        avg_struct = self.load_avg_structure()
        
        n_frames = u.trajectory.n_frames
        
        # Split frame indices evenly among workers.
        frame_chunks = np.array_split(range(n_frames), self.n_jobs)
        
        avg_pdb_name = self.avg_pdb_name

        #DO NOT CHANGE SELECTION WITHOUT CHANGING ALIGNMENT, AVERAGING AND LOOKING AT DISTANCE MATRIX FUNCTION
        selection = "protein and name CA and not resname ACE NME"

        #TODO change parallelism scheme here
        with mp.Pool(self.n_jobs) as pool:
            results = pool.starmap(
                analysis.compute_rmsf_chunk,
                [(self.topology, self.traj_file, list(chunk), selection, avg_pdb_name)
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
        u_out.trajectory[-1]  # Go to the last frame

        if colored_pdb_ending is not None:
            u_out.atoms.write(f"{self.basename}{colored_pdb_ending}")
        elif colored_pdb_ending is None:
            u_out.atoms.write(self.pdb_file) 

        #plot the RMSF values
        if self.plot:
            protein = u.select_atoms(selection)
            residue_list = range(1, len(rmsf_values) +1)
            plot.line_plotter_2d(residue_list, rmsf_values, "residue", "RMSF (Å)", 
                                 self.basename, "1d_rmsf", force_zero_start_x=True, 
                                 force_zero_start_y=True, plot_format=self.plot_format)

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

        u = self.u

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
                plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df[rmsd_columns], "Time (ps)", "RMSD (Å)", 
                                     self.basename, "1d_rmsd", force_zero_start_x=True, 
                                     force_zero_start_y=True, plot_format=self.plot_format)
                
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

        u = self.u

        atomgroup = u.select_atoms(selection)
        rga = mda.analysis.base.AnalysisFromFunction(analysis.radgyr, u.trajectory, atomgroup, atomgroup.masses, total_mass=np.sum(atomgroup.masses)).run(backend="multiprocessing", n_workers= self.n_jobs)

        rg_labels = [f"Rg_all_{legend}", f"Rg_x_{legend}", f"Rg_y_{legend}", f"Rg_z_{legend}"]
        rg_df = pd.DataFrame(rga.results.timeseries, columns=rg_labels)
        self.time_df = pd.concat([self.time_df, rg_df], axis=1)

        if self.plot:
            plot.line_plotter_2d(self.time_df["Time (ps)"], self.time_df[rg_labels], "Time (ps)", "Radius of gyration (Å)", 
                                 self.basename, "rg", force_zero_start_x = True, histogram = True, plot_format=self.plot_format)

        print(f"{round(time.time() - start_time,2)}s used for rgyr")

    
    def write_sparse_traj(self, start_frame = 0, end_frame = -1):
        """ 
        Writes a sparse trajectory for analysis.
        Useful for parallel_2d_rmsd as the scaling is N^2
        Sparsity is currentlly calculated so that the sparse traj will at most have self.max_sparse_frames
        Saves the sparse trajectory to self.basename_sparse.dcd
        
        Parameters:
            start_frame : int
                The frame at which to start printing the trajectory
            end_frame : int
                The frame at which to stop printing the trajectory

        Returns:
            str : name of the sparse trajectory
        """
        
        #calculate total frames in query 
        u = self.u
        tot_frames = u.trajectory.n_frames

        if end_frame == -1 or end_frame >= tot_frames -1 :
            end_frame = tot_frames-1

        #this specifies the size, start and end of the 2D RMSD matrix
        #BEWARE N^2 scaling operation
        sparsity = (tot_frames - (tot_frames - end_frame) - start_frame)/ self.max_sparse_frames
        
        #Round up and convert to int by exploiting floating point remainder
        sparsity = int(sparsity // 1 + (sparsity % 1 > 0))

        #if we round down to 0 we round back up
        if sparsity == 0:
            sparsity = 1

        #add args to class variables
        self.sparsity = sparsity
        self.sparse_traj_start_frame = start_frame
        self.sparse_traj_end_frame = end_frame
        
        self.sparse_traj = f"{self.basename}_trajectory_sparse.dcd"
        #endframe+1 due to noninclusive slicing
        self.sparse_time_ps = utils.write_trajectory(u, self.sparse_traj, sparsity=sparsity, start_frame=start_frame, end_frame=end_frame+1)

        print(f"\nwrote sparse traj from frame {start_frame} to {end_frame} with sparsity {sparsity}, one frame per {sparsity * self.reporting_time} ps, for a total of {len(self.sparse_time_ps)} frames \n")
        
        #quick check for nasty behavior
        sparse_u = mda.Universe(self.sparse_traj)
        assert len(self.sparse_time_ps) == sparse_u.trajectory.n_frames, "Mismatch between sparse_time_ps and sparse trajectory frames"
        
        return self.sparse_traj
    
    def remove_sparse_traj(self):
        """
        Removes the sparse traj printed for 2D RMSD analysis
        """

        #remove temp sparse traj
        os.remove(self.sparse_traj)

        #reset class variables
        self.sparse_traj = None
        self.sparse_traj_start_frame = None
        self.sparse_traj_end_frame = None
        self.sparsity = None
        self.sparse_time_ps = None
        print("removed sparse trajectory from disk")
        
    
    def calc_2d_rmsd(self, selection_list=["backbone"], legend_list = None):
        """ 
        Compute the full 2D RMSD matrix efficiently using process-based parallelism.
        runs write_sparse_traj() to write at most self.max_sparse_frames
        if self.sparse_traj is not set and then removes the sparse trajectory.
        If self.sparse_traj is set this function does not remove the sparse traj.

        mdaencores parallelized distance matrix scheme was tried here but was found to not parallelize properlly.
        This function does a reference calculation in 5s while it takes mdaencore 50s on the same system, both calculations with 22 cores.
        
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

        #check that user input was valid if using manual legend
        if legend_list is None:
            legend_list = utils.strip_mda_selection(selection_list)

        elif len(selection_list) != len(legend_list):
            raise ValueError("selection list did not match the legend list in 2D RMSD calculation")

        #start the results dict
        rmsd_matrix_dict = {}

        #iterate over calculations
        for selection, legend in zip(selection_list, legend_list):

            with mp.Pool(self.n_jobs) as pool:
                #TODO revise all parallelization here and in RMSF to start more reliablly
                results_list = pool.starmap(analysis.compute_2d_rmsd_block, [(self.topology, self.sparse_traj, selection, list(pairs)) for pairs in split_pairs])

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
                plot.time_heatmap(rmsd_matrix, "Time (ps)", "Time (ps)", "RMSD (Å)", f"2D RMSD for {legend}", f"2d_rmsd_{legend}", 
                                  self.basename, self.reporting_time, self.sparsity, 
                                  self.sparse_traj_start_frame, plot_format=self.plot_format)

        #if the user did not start a sparse traj remove the automatically created one
        if delete:
            self.remove_sparse_traj()


        print(f"{round(time.time() - start_time,2)}s used for 2D RMSD")

        return legend_list, rmsd_matrix_dict


    def calc_correlation_matrix(self, selection = "name CA and not resname ACE NME"):
        """
        Calculates the correlation matrix for protein movements for the selection.
        Can be used with selection = "name CA and not resname ACE NME" to generate input matrix for Silvia Osunas SPM method.
        This method is not parallelized since it is already very fast.
        universe blocking for parallelization might be implemented in the future.

        Parameters:
            selection: str
                The selection for which the correlation matrix is calculated
        Returns:
            np.ndarray : The correlation matrix of size (N, N) where N is the number of atoms in the selection
        """

        print("computing covariance matrix")
        start_time = time.time()

        u = self.u

        # Select atoms based on the provided selection string and calc covariance matrix
        # produces a 3N x 3N matrix where N is the number of atoms in the selection
        covariance_marix = mdaencore.covariance.covariance_matrix(u, select=selection)

        correlation_matrix = analysis.covariance_to_correlation(covariance_marix)
        correlation_matrix = analysis.matrix_3M_to_M(correlation_matrix)

        #save correlation matrix as npy file
        np.save(f"{self.basename}_correlation_matrix.npy", correlation_matrix.astype(np.float32))

        if self.plot:
            res_nr_minmax = (1, correlation_matrix.shape[0])
            sel_strip_name = utils.strip_mda_selection(selection_string=selection)

            plot.heatmap(correlation_matrix, x_axis_min_max=res_nr_minmax, y_axis_min_max=res_nr_minmax, 
                         x_var="Residue number", y_var = "Residue number", heat_var = "correlation",
                         titel=f"Correlation matrix for {utils.strip_mda_selection(selection, spaces_to_underscores=False)}", 
                         plot_type= f"correlation_mat_{sel_strip_name}", basename = self.basename, plot_format=self.plot_format)

        print(f"{round(time.time() - start_time,2)}s used for correlation matrix")

        return correlation_matrix
        
    def calc_distance_matrix(self, structure ="AVG", selection = "name CA and not resname ACE NME"):
        """
        Calculates the distance matrix between two selections in the average structure.

        THIS IS NOT A TIME AVERAGED DISTANCE MATRIX, IT IS THE DISTANCE MATRIX OF THE AVERAGE STRUCTURE.
        The distance matrix is also NOT a RMSD matrix but a pairwise distance matrix for all atoms in the selection.

        Can be used with selection = "name CA and not resname ACE NME" to generate input matrix for Silvia Osunas SPM method.
        This method is not parallelized since it is already very fast.
        universe blocking for parallelization might be implemented in the future.

        Parameters:
            structure : str
                The structure to use for distance matrix calculation. Supported options are:
                "AVG" : The average structure (suitable for general distance matrix analysis)
                filename : A string pointing to a PDB file to be used for distance matrix calculation
            selection: str
                The selection for which the atom to atom distance matrix is calculated

        Returns:
            np.ndarray : The distance matrix
        """

        print("computing distance matrix")
        start_time = time.time()

        if structure.upper() == "AVG":
            u = self.load_avg_structure()
        
        else:
            if os.path.exists(structure) == False:
                raise ValueError(f"structure file {structure} does not exist for distance matrix calculation")
            u = mda.Universe(structure)

        # Select atoms based on the provided selection strings
        atomgroup = u.select_atoms(selection)

        distance_matrix = mda.analysis.distances.distance_array(atomgroup, atomgroup)

        sel_strip_name = utils.strip_mda_selection(selection_string=selection)

        #save distance matrix as npy file
        np.save(f"{self.basename}_distance_matrix_{sel_strip_name}.npy", distance_matrix.astype(np.float32))

        if self.plot:
            plot_minmax = (1, distance_matrix.shape[0])

            plot.heatmap(distance_matrix, x_axis_min_max=plot_minmax, y_axis_min_max=plot_minmax, 
                         x_var="Residue number", y_var = "Residue number", heat_var = "distance (Å)",
                         titel=f"Distance matrix for {utils.strip_mda_selection(selection, spaces_to_underscores=False)}", 
                         plot_type= f"distance_mat_{sel_strip_name}", 
                         basename = self.basename, aspect= "equal", plot_format=self.plot_format)

        print(f"{round(time.time() - start_time,2)}s used for distance matrix")

        return distance_matrix

    def make_mdaencore_triangle(self, selection, legend = None):
        """ 
        This function automatically handles the logic behind reading/reusing/creating a mdaencore TriangularMatrix.
        If a 2D RMSD matrix for the selection exists on disk it with its sparse traj, this is read and 
        converted to a TriangularMatrix.
        If not a new 2D RMSD calculation is commisioned and the resulting matrix and sparse traj is used to create the TriangularMatrix.

        Parameters:
            selection : str
                The selection for which the TriangularMatrix is to be created.
            legend : str
                The stripped selection name used for naming the 2D RMSD matrix file.
                If left at none the selection string is stripped and used.
                This conforms with default behavior in calc_2d_rmsd()
        Returns:
            mdaencore.utils.TriangularMatrix : The triangular matrix for the selection
            delete : bool
                Whether the sparse traj created for this operation should be deleted after use.
        """
        delete = False

        if legend is None:
            legend = utils.strip_mda_selection(selection)

        if os.path.exists(f"{self.basename}_2d_rmsd_{legend}.csv") and self.sparse_traj is not None:
            print(f"reading existing 2D RMSD matrix for mdatrianglematrix for {legend}")
            rmsd_array = pd.read_csv(f"{self.basename}_2d_rmsd_{legend}.csv", index_col=0).to_numpy()

        else:
            if os.path.exists(f"{self.basename}_2d_rmsd_{legend}.csv") == False: 
                print(f"no existing 2D RMSD matrix found, calculating new one for mdatrianglematrix for {legend}")
            elif self.sparse_traj is None:
                print(f"no sparse trajectory found, writing a new one and calculating new 2D RMSD matrix for mdatrianglematrix for {legend}")
            
            #write sparse traj here so 2D RMSD does not delete it afterwards
            self.write_sparse_traj()
            delete = True

            legend_list, rmsd_dict = self.calc_2d_rmsd(selection_list=[selection])
            rmsd_array = rmsd_dict[legend_list[0]]
        
        #Flatten the lower triangle of the RMSD array to make it compatible with mdaencore
        lower_triangle_array = rmsd_array[np.tril_indices(rmsd_array.shape[0])]

        # Instantiate a mdaencore triangular matrix from the rmsd array
        rmsd_triangle = mdaencore.utils.TriangularMatrix(lower_triangle_array)

        return rmsd_triangle, delete
    
    def cluster_trajectory(self, selection = "backbone", method_name = "HDBSCAN", min_cluster_size=5, eps = 1, min_samples=5,  n_clusters = 5):
        """ 
        Clusters a trajectory using either DBSCAN or KMeans from mdaencore depending on what method is selected.

        This analysis will try to use the precomputed 2D RMSD matrix and its accompanying sparse trajectory
        if availible. If not, it will write a new sparse traj, compute a new 2D RMSD matrix and remove
        the printed sparse trajectory.

        Parameters:
            selection : str
                The selection to be used for clustering.
            method_name : str
                The clustering method to be used. Either "HDBSCAN", "DBSCAN" or "KMEANS". Not case sensitive
                DBSCAN and KMEANS are from mdaencore. HDBScan is a custom implementation in whatcat using sklearns HDBSCAN
                and som fancy trickery to make it work with mdaencore clustering framework.
                Recommended method is HDBSCAN as it does not require predefining number of clusters or epsilons while still
                being robust in clustering actual MD trajectories in our hands.
            min_cluster_size : int
                The min_cluster_size parameter for HDBSCAN clustering. Also sets the min_samples parameter for HDBSCAN internally.
            eps : float
                The eps parameter for DBSCAN clustering. 
                eps is the RMSD difference we allow at most to say two frames are core connected
            min_samples : int
                The min_samples parameter for DBSCAN clustering.
            n_clusters : int
                The number of clusters to produce with KMeans clustering.
                
        Returns:
            str : The filepath to the PDB containing the cluster centroids ordered with the most populated first
            pd.DataFrame : Dataframe containing information about the clustering results
        """
        
        #Set starting variables
        start_time = time.time()
        legend = utils.strip_mda_selection(selection)

        #Sanitize user input
        method_name = method_name.upper()

        print(f"clustering trajectory for {legend}")

        #make or read RMSD triangle matrix for clustering
        rmsd_triangle, delete = self.make_mdaencore_triangle(selection)

        #Start the mda Universe
        sparse_u = mda.Universe(self.topology, self.sparse_traj)

        #Set the method based on the user input
        if method_name == "KMEANS":
            method = mdaencore.clustering.ClusteringMethod.KMeans(n_clusters)
            threshold = n_clusters
        
        elif method_name == "DBSCAN":
            method = mdaencore.clustering.ClusteringMethod.DBSCAN(eps=eps, min_samples=min_samples)
            threshold = f"{eps}_{min_samples}"
        
        elif method_name == "HDBSCAN":
            method = analysis.HDBscan_mdaencore(min_cluster_size=min_cluster_size)
            threshold = f"{min_cluster_size}"
        
        # Get the first (and only since we provide only one universe) clustercollection
        clusters= mdaencore.cluster(sparse_u, select=selection, method = method, distance_matrix=rmsd_triangle, 
                                                       allow_collapsed_result=False, ncores=-1)[0] 

         #write a DCD containing all clusters in order of number of elements
        cluster_frames = []
        cluster_no = 1
        clusters_min_max = {}

        #Also save to a pandas dataframe
        cluster_df = pd.DataFrame(columns=["Cluster_ID", "N_frames", "Time_sampled (ps)", "Centroid_frame", "Centroid_time (ps)"])

        time_points = self.sparse_time_ps
        unit = "ps"
        if max(time_points) > 2000:
            time_points = [time_point /1000 for time_point in time_points]
            unit = "ns"

        for cluster in sorted(clusters.clusters, key=lambda c: len(c.elements), reverse=True):
            #Save cluster centroid for writing to DCD
            cluster_frames.append(cluster.centroid)

            #Save contigous ranges for plotting
            contigous_frames = analysis.list_contigous_range(cluster.elements)
            clusters_min_max[cluster_no] = [(time_points[start], time_points[end]) for start, end in contigous_frames] #convert frame indices to time(ps)
            
            #Calc frame of centroid in orginal traj
            #+1 to convert from 0 indexing to the 1 indexing used in visuallization software
            frame_nonsparse = cluster.centroid * self.sparsity + self.sparse_traj_start_frame +1

            sampled_time = len(cluster.elements) * self.sparsity * self.reporting_time 
            centroid_time = frame_nonsparse * self.reporting_time
            
            #Write to dataframe
            temp_df = pd.DataFrame({"Cluster_ID": [cluster_no],
                                    "N_frames": [len(cluster.elements) * self.sparsity],
                                    "Time_sampled (ps)": [sampled_time],
                                    "Centroid_frame": [frame_nonsparse],
                                    "Centroid_time (ps)": [centroid_time]})
            cluster_df = pd.concat([cluster_df, temp_df], axis=0)

            cluster_no += 1

        # Write the cluster centroids to a new DCD file
        clustered_traj_file = f"{self.basename}_clustered_{legend}_{method_name}_{threshold}.dcd"
        with mda.Writer(clustered_traj_file, sparse_u.atoms.n_atoms) as writer:
            for ts in sparse_u.trajectory[cluster_frames]:
                writer.write(sparse_u.atoms)

        print(f"Cluster centroids written to {clustered_traj_file}")
        
        #write the cluster dataframe to csv
        cluster_df.to_csv(f"{self.basename}_clustered_{legend}_{method_name}_{threshold}.csv", index=False)

        #If we want to plot the cluster membership over time we do it here
        if self.plot:
            #broken bar plotter lives here since there is no application for it outside clustering right now
            #This might be subject to change in the future TODO utils.plotting module?

            # Create a two-panel figure: broken barh on the left, square heatmap on the right
            fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})

            for cluster_id, min_max_list in clusters_min_max.items():
                width_unit = time_points[1] - time_points[0] 
                #Convert min,max to min,width for broken_barh
                #We add 1 width units one each side to make the bars touch each other and to show clusters which are only one frame long
                min_width_list = [(start - width_unit*1 , end - start + width_unit*1) for start, end in min_max_list]
                ax_bar.broken_barh(min_width_list, (cluster_id -0.4, 0.8))

            ax_bar.set_yticks(range(1, len(clusters.clusters) +1), labels=clusters_min_max.keys())
            ax_bar.set_xlim(time_points[0], time_points[-1])
            ax_bar.set_xlabel(f"Time ({unit})")
            ax_bar.set_ylabel("Cluster ID")
            ax_bar.set_title(f"Cluster membership over time for {legend} using {method_name} clustering")

            #Calculate centroid 2D RMSD matrix for heatmap
            centroid_u = mda.Universe(self.topology, clustered_traj_file)

            #calc 2D RMSD matrix but dont superimpose as centroids were aligned during centering and alignment of traj
            cluster_rmsd = mdaencore.confdistmatrix.get_distance_matrix(centroid_u, select=selection, superimpose=False, n_jobs = -1).as_array()

            ax_heat.imshow(cluster_rmsd, cmap='viridis', interpolation='nearest')
            ax_heat.set_title(f"Centroid RMSD matrix for {legend} using {method_name} clustering")

            #correct labels for 1 indexing
            ax_heat.set_xticks(range(len(clusters.clusters)), labels=range(1, len(clusters.clusters) +1))
            ax_heat.set_yticks(range(len(clusters.clusters)), labels=range(1, len(clusters.clusters) +1))

            ax_heat.set_xlabel("Cluster Centroid Index")
            ax_heat.set_ylabel("Cluster Centroid Index")
            plt.colorbar(ax_heat.images[0], ax=ax_heat, label="RMSD (Å)", fraction=0.046, pad=0.04)

            plt.savefig(f"{self.basename}_clusterplot_{legend}_{method_name}_{threshold}.{self.plot_format}")

        #If we wrote a sparse traj automatically we remove it
        #The user gets a permanent sparse traj if they request it explicitlly
        if delete:
            self.remove_sparse_traj()

        print(f"{round(time.time() - start_time,2)}s used for clustering")

        return clustered_traj_file, cluster_df

    def dimensionality_reduction (self, method_name = "PCA", selection = "name CA", n_components = 3):
        """ 
        Performs dimensionality reduction using PCA from mdaencore
        
        Parameters:
            method_name : str
                The dimensionality reduction method to be used. 
                "PCA" uses sklearn principal component analysis while "AP" uses affinity propatgation from mdaencore
                Not case sensitive
            selection : str
                The selection to be used for dimensionality reduction.
            n_components : int
                The number of components to reduce to.

        Returns:
            np.ndarray : The reduced data of shape (n_frames, n_components)
        """
        
        #Set starting variables
        start_time = time.time()
        legend = utils.strip_mda_selection(selection)

        #Sanitize user input
        method_name = method_name.upper()

        print(f"Dimensionality reduction for {legend}")

        #make or read RMSD triangle matrix for clustering
        rmsd_triangle, delete = self.make_mdaencore_triangle(selection)

        #Start the mda Universe
        sparse_u = mda.Universe(self.topology, self.sparse_traj)

        if method_name == "PCA":
            method = mdaencore.PrincipalComponentAnalysis(dimension = n_components)
        elif method_name == "AP":
            method = mdaencore.StochasticProximityEmbeddingNative(dimension=n_components)

        #Calculate reduced dimension coordinate
        #Returns one coordinate array (n_components, n_frames) and a dict with key "ensamble membership" containing cluster membership if applicable
        #which it is not here
        red_dim, ensamble_membership = mdaencore.reduce_dimensionality(sparse_u, method = method, select=selection, 
                                              distance_matrix = rmsd_triangle, ncores = self.n_jobs)

        red_dim_df = pd.DataFrame(red_dim.transpose(), columns=[f"{method_name} dim {i+1}" for i in range(n_components)])

        if self.plot:
            plot.line_plotter_2d(self.sparse_time_ps, red_dim_df, f"Time (ps)", 
                                 f"{method_name} of {legend}", basename=self.basename, plot_type=f"PCA_{legend}", 
                                 plot_format=self.plot_format)

        #If we wrote a sparse traj automatically we remove it
        #The user gets a permanent sparse traj if they request it explicitlly
        if delete:
            self.remove_sparse_traj()

        print(f"{round(time.time() - start_time,2)}s used for dimensionality reduction")

    
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
    
    def run_prolif(self, analysis_resnames, analyze_water = False, start_ps = 0, stop_ps = None, sparsity = 1):
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
                At what point in the traj shall we stop analysis? None means we run the entire trajectory.
                Remember slicing is noninclusive so stop_ps frame is not analyzed.
            sparsity: int
                How often do we sample the trajectory for snapshots to be analyzed?

        Returns:
            tuple : list of dictonary keys, dictionary of interaction dataframes
        """

        print("Computing interaction fingerprints")
        start_time = time.time()

        # load universe from self
        u = self.u

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
            #We pass force as kwarg to allow the analysis of only metals which do not have hydrogens hence breaking RDKit
            ligand_mol = prolif.Molecule.from_mda(ligand_selection, force=True)

            # use default interactions + metal interactions
            fp = prolif.Fingerprint(interactions = [
                                    'MetalAcceptor', 'MetalDonor', 'Hydrophobic',
                                    'HBAcceptor', 'HBDonor', 'Cationic', 'Anionic',
                                    'CationPi', 'PiCation', 'PiStacking', 'VdWContact',
                                    ], count=True)

            #calculate frame starts and ends
            start_frame = round(start_ps/self.reporting_time)

            if stop_ps == None:
                stop_frame = None
            else:
                stop_frame = round(stop_ps/self.reporting_time)

            # run on a slice of the trajectory frames: from start to stop with a step of sparsity
            #We pass force as kwarg to allow the analysis of only metals which do not have hydrogens hence breaking RDKit
            
            #Due to Prolif not handling mda.Frameiteratorall we have to special case the full traj with no sparsity
            #TODO issue filed, prolif #322
            if start_frame == 0 and stop_frame == None and sparsity == 1:
                fp.run(u.trajectory, ligand_selection, protein_selection, n_jobs = self.n_jobs, converter_kwargs=({"force": True}, {}))
            #For all other cases we slice the trajectory
            else:
                fp.run(u.trajectory[start_frame:stop_frame:sparsity], ligand_selection, protein_selection, n_jobs = self.n_jobs, converter_kwargs=({"force": True}, {}))
            
            if self.plot:
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
                ax.set_title(f"Interaction barcode for {utils.strip_mda_selection(ligand_selector_string)}")
                # Save barcode to PNG
                ax.figure.savefig(f"{self.basename}_{utils.strip_mda_selection(ligand_selector_string)}_prolif_barcode.{self.plot_format}", dpi=300, bbox_inches="tight")
            
            #make prolif df
            interaction_df = fp.to_dataframe()

            # Insert Time (ps) as the first column in the DataFrame
            frame_times = [ts.time for ts in u.trajectory[start_frame:stop_frame:sparsity]]
            interaction_df.insert(0, "Time (ps)", frame_times)
            interaction_df.to_csv(f"{self.basename}_{utils.strip_mda_selection(ligand_selector_string)}_prolif_df.csv", index=False)

            # Save ligand network plots at different occurrence thresholds
            for threshold in [0.10, 0.30, 0.50, 0.90]:
                try:
                    lignetwork = fp.plot_lignetwork(ligand_mol, threshold=threshold)
                    # fp.plot_lignetwork returns an IPython.display.HTML object, but the HTML content is in lignetwork.data
                    with open(f"{self.basename}_{utils.strip_mda_selection(ligand_selector_string)}_lignetwork_{threshold}.html", "w") as f:
                        f.write(lignetwork.data)
                except:
                    print(f"No interactions found for {utils.strip_mda_selection(ligand_selector_string)} satisfying threshold {threshold}, skipping lignetwork plot.")

            # Tanimoto similarity matrix
            bitvectors = fp.to_bitvectors()
            similarity_matrix = []
            for bv in bitvectors:
                similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(bv, bitvectors))
            similarity_matrix = pd.DataFrame(similarity_matrix, index=interaction_df.index, columns=interaction_df.index)

            if self.plot:
                plot.time_heatmap(similarity_matrix, x_var="Time (ps)", y_var="Time (ps)", 
                                  heat_var="Tanimoto similarity", titel="Binding pose similarity", 
                                  plot_type=f"{utils.strip_mda_selection(ligand_selector_string)}_prolif_2d",  
                                  basename=self.basename, reporting_time =self.reporting_time, sparsity = 1, 
                                  start_frame = 0, plot_format=self.plot_format)

            interaction_df_dict[ligand_selector_string] = interaction_df
        
        print(f"{round(time.time() - start_time,2)}s used for interaction fingerprints")

        return list(interaction_df_dict.keys()), interaction_df_dict
    
    def calc_efield(self, scan_resid_sel, efield_vec_sel, sparse=True, method="amoeba_mutual", run_decomp=False, run_pointcharge=False):
        """
        Runs the electric field analysis on the trajectory.
        This method is very general and does not leverage the full range of options presented in the efield
        module. For advanced calculations the user is encouraged to write a purpose specific wrapper.

        Parameters:
            scan_resid_sel: mdanalysis selection str
                Selection string for the residue whoose interactions you want to monitor.
            efield_vec_sel: tuple of mdanalysis selection str
                2 member tuple of selections yielding 1 or more atoms each who define the approximate 
                transition state dipole moment. The fictional particle will be added halfway between these.
                By convention the tuple should be (positive charge, negative charge) in the TS.
                This means all stabilizing fields will have a negative sign.
            sparse: bool
                Whether to run on the sparse trajectory or the full trajectory.
                Running decomp on the full trajectory is VERY expensive and not recommended
            method: str
                Which forcefield/polarization scheme to use for the calculation, see below for options
                For a reference workload the relative runtimes traj R^2 and decomp R^2 (amoeba mutual as reference) were
                    "amber"                3%, 0.60, 0.72
                    "amoeba_direct"       59%, 0.74, 0.28
                    "amoeba_extrapolated" 79%, 0.996, 0.999
                    "amoeba_mutual"      100%, 1, 1
            run_decomp: bool
                Whether to run electric field decomposition to, in addition to total electric field,
                also report which amino acids give rise to each field.
            run_pointcharge: bool
                If we want to run the pointcharge decomposition to see what the effect would be of introducing a charge
                at Cb of a residue.

        Returns:
            np.ndarray : The array containing the projected electric field over the relevant trajectory.
                This is also written to the time dataframes.
            The electric field decomposition and pointcharge field is written to the residue dataframe

        """
        print("Computing electric fields")
        start_time = time.time()

        #make sure we have a sparse traj if requested
        delete = False
        if sparse:
            if self.sparse_traj is None:
                self.write_sparse_traj()
                delete = True
            else:
                trajectory = self.sparse_traj

        #if using full trajectory
        else:
            trajectory = self.traj_file

        extra_decomp_residue_sel = ["name MG", "name ZN"]

        if method.upper() == "AMBER":
            efield_calc = efield.Efield_amber(self.topology, trajectory, scan_resid_sel, 
                                              efield_vec_sel, self.platform, extra_decomp_residue_sel=extra_decomp_residue_sel)
        
        elif method.upper().split("_")[0] == "AMOEBA":
            efield_calc = efield.Efield_amoeba(self.topology, trajectory, scan_resid_sel, efield_vec_sel, 
                                               polarization_type=method.upper().split("_")[1], platform=self.platform,
                                               extra_decomp_residue_sel=extra_decomp_residue_sel)
        else:
            raise ValueError("invalid method input")
        
        efield_traj = efield_calc.get_field_trajectory()
        sparse_df = pd.DataFrame({"Time (ps)":self.sparse_time_ps, f"efield_{scan_resid_sel.replace(" ", "_")}":efield_traj})
        self.time_df = pd.merge(self.time_df, sparse_df, how="left", on="Time (ps)")

        if self.plot:
            plot.line_plotter_2d(self.sparse_time_ps, efield_traj, "Time (ps)",
                             "Efield (kJ/mol*debye)", self.basename, plot_type="efield_traj", plot_format=self.plot_format,
                              histogram=True )

        print(f"{round(time.time() - start_time,2)}s used for electric field over traj")

        if run_decomp:
            #amoeba_mutual: 86 ms/frame on work laptop with power cord
            #4.7 h for 400 residues decomposed for 500 frames
            start_time = time.time()
            efield_decomp = efield_calc.get_field_decomposed()
            self.residue_df = pd.merge(self.residue_df, efield_decomp, how="outer")
            
            if self.plot:
                plot.scatter_plot_2d(efield_decomp["resid_idx"]+1, efield_decomp["dfield_avg"], "residue number",
                                 "Efield kJ/(mol*debye)", self.basename, "efield_resid", self.plot_format,
                                 "Electric field decomposition", 
                                 labels=efield_decomp["resname"] + efield_decomp["resid_idx"].astype(str))

            print(f"{round(time.time() - start_time,2)}s used for electric field decomposition")
        
        if run_pointcharge:
            start_time = time.time()
            efield_pointcharge = efield_calc.get_pointcharge_field()
            self.residue_df = pd.concat([self.residue_df, pd.DataFrame({"pointcharge_field":efield_pointcharge})], axis=1)
            
            print(f"{round(time.time() - start_time,2)}s used for pointcharge electric field over traj")
        
        #Plot decomp and pointcharge field as a 1D and/or 2D plot
        if run_decomp or run_pointcharge:
            # Colour structure by dfield_avg
            u_out = mda.Universe(self.topology, self.traj_file)
            u_out.add_TopologyAttr('tempfactors')
            u_out.add_TopologyAttr('occupancies')
            
            #handle metal ion selection
            if len(extra_decomp_residue_sel) == 0:
                selection_str = "protein"
            else:
                selection_str = "protein or " + " or ".join(extra_decomp_residue_sel)

            protein_and_others = u_out.select_atoms(selection_str)
            protein = u_out.select_atoms("protein")

            if run_decomp and run_pointcharge and self.plot:
                #plot the 2D scatterplot
                labels = protein.residues.resnames + (protein.residues.ix_array+1).astype(str)
                plt = plot.scatter_plot_2d(efield_pointcharge, efield_decomp["dfield_avg"][0:len(efield_pointcharge)], "Cb pointcharge field kJ/(mol*debye)", 
                                     "actual electric field (kJ/mol*debye)", titel="actual vs predicted Efield", basename=self.basename, plot_type="efield_2D",
                                     plot_format=self.plot_format, labels=labels, save_fig=False, histogram=False)
                plt.axhline(0, color="black", linestyle="-", linewidth=0.8)  # Horizontal line at y=0
                plt.axvline(0, color="black", linestyle="-", linewidth=0.8)  # Vertical line at x=0
                plt.savefig(f"{self.basename}_efield_2d.{self.plot_format}")

            #Start preparing mda to print a PDB with efield info in it
            if run_decomp:
                # Assign computed dfield_avg values to residues
                for residue, dfield_avg in zip(protein_and_others.residues, efield_decomp["dfield_avg"]):
                    residue.atoms.tempfactors = dfield_avg

            if run_pointcharge:
                # Assign computed dfield_avg values to residues
                for residue, dfield_pointcharge in zip(protein.residues, efield_pointcharge):
                    residue.atoms.occupancies = dfield_pointcharge

            #write pdb 
            u_out.trajectory[-1]  # Go to the last frame
            u_out.atoms.write(f"{self.basename}_efield.pdb")

        #Clean up the efield trajectory file
        efield_calc.delete_traj()

        if delete:
            self.remove_sparse_traj()
            
    
    def save_df_to_csv(self):
        """
        saves time_df and residue_df to csv

        Parameters:
            nothing

        Returns:
            nothing
        """
        print("saving dataframes to csv")
        self.time_df.to_csv(f"{self.basename}_time.csv", index=False)
        self.residue_df.to_csv(f"{self.basename}_residue.csv", index=False)
        print("saved dataframes to csv")


def main():
    """
    Main function to run the MD simulation and analysis.
    """

    #Start the command line parser
    parser = argparse.ArgumentParser(
                        prog="whatcat-md",
                        description=(
                        "This script sets up a OPENMM  simulation of a protein (amber ff14SB)," 
                        "any small molecules/cofactors (Sage 2.2.1) as well as water/ions using a 12-6 model (amber ff14/tip3pfb)."
                        "Simulations are ran using 1 nm of padding with enough ions to neutralize the system."
                        "Known issues:"
                        "If you get a template error on terminal residue, you probably have a bad chain termination in your PDB. "
                        "C terminal errors can be fixed by running ChimeraX dockprep"),
                        epilog="Use with care and acknowledge Erik Sundén and the Per-Olof Syrén group at KTH Sweden")

    parser.add_argument("pdb", type = str, help = "PDB structure of the structure you want to simulate. \nWARNING PDB may not contain any ligands. These must be provided from sdf files") 
    parser.add_argument("-l", "--lig", type = str, action="append", default = [], help = ("optional parameter, SDF file containing all nonstandard ligands and cofactors." 
                                                                        "Convenientlly produced by drawing in chemdraw and exporting as SDF then docking with added hydrogens."
                                                                        "This is easilly done by checking ChimeraX dockpreps charge assignment when running dockprep."
                                                                        "WARNING charges MUST be assigned in the sdf file, use -cc True to autoassign based on pH"
                                                                        "or whatcat/md/molecule_inspector.ipynb which both converts files and visuallizes result")) 
    parser.add_argument("-restart", "--restart", type = str, default= "False", choices=["true", "True", "false", "False"], help="Restarts the simulation from restart xml files if set to True \nRequieres that pdb is set to pdbname_final.pdb", required=False)
    parser.add_argument("--platform", type = str, default= "CUDA", help="""Sets the simulation platform and optionally device if specified as a comma separated string of ints eg "CUDA_0" or "CUDA_0,1", default = "CUDA" """, required=False)

    parser.add_argument("--pdbfixer", type = int, default = 2, help = ("0, 1, 2 depending on if your structure shall be PDBfixed." 
                                                                        "2=default removes and readds hydrogens as well as tries to find missing atoms"
                                                                        "good if you have a SEQRES and unresolved loops as well as unhandled disulfide bonds."
                                                                        "=1 fixes loops and so on but retains hydrogens in structure. Good if manual protonation was done"
                                                                        "=0 does not fix your pdb, make sure it is good and residue names are labelled correctlly after protonation state eg CYX for deprotonated CYS" ))
    parser.add_argument("-cc", "--charge_correct", type = str, default= "False", choices=["true", "True", "false", "False"], 
                        help="""Whether to charge correct ligands or not. also converts files to sdf if ligand not sdf.
                                If False we take the charges in the SDF file and protonate accordinglly.
                                If True we look at the protonation in the sdf file and guess the charges. 
                                total charge guessed is printed for user inspection""")
    parser.add_argument("--variants", type = str, action="append", default= [], help ="""A comma separated str of residue index, resname. Eg to change Cys120 to CYX you would provide "120,CYX".
                        To specify multiple variants you can use several --variants parameters""", required=False)
    parser.add_argument("--solvate", type = int, default= 2, choices=[0,1,2], help="Regulates solvation. \ndefault = 2 - remove all water and add a solvent box \n 1 = add solvent box \n 0 = do not alter solvent, solvation box must be provided by user", required=False)
    parser.add_argument("-pbc", type = str, default= "True", choices=["true", "True", "false", "False"], help="Changes whether or not we create the simulation with PBC. " \
        "Can be useful to set to False when creating topologies for QM/MM in implicit solvation. Using this setting requieres that your PDB does not have a cryst record.", required=False)
    parser.add_argument("-ph", "--ph", type = float, default= 7.4, help="Sets pH for the simulation using PDBfixer and if using -cc openbabel. Default pH = 7.4")

    parser.add_argument("-consth","--constrain_h", type = str, default= "True", choices=["true", "True", "false", "False"], 
                        help="Whether to run a constrain H-heavy bonds and use HMR (default, dt stable up to 5 fs) or run hydrogens (including those on water) like other atoms (dt stable up to ca 0.5 fs)."
                        "Must be false to use restart_system.xml as input for Orca FF", required=False)

    parser.add_argument("-t", "--timeprod", type = float, default= 1, help="Production simulation time in ns. Accepts floats and ints")
    parser.add_argument("-rt", "--report_time", type = float, default= 10, help="Reporting frequency in ps")
    parser.add_argument("-eqt", "--equillibration_time", type = float, default= 50, help="Equillibration time in ps, do not set lower than 50 ps. \nUsed for both NPT and NVT equillibration")
    parser.add_argument("-min", "--minimize", type = str, default= "True", choices=["true", "True", "false", "False"], help="do you want to minimize the structure before equillibration? Default True", required=False)
    parser.add_argument("-dt", "--timestep", type = float, default= 4, help="Simulation timestep in fs. Accepts ints. Default is 4 fs which works well for most applications")
    
    parser.add_argument("--resname", type = str, action="append", default= [], help="Residue names in PDB for which you want further analysis, eg ligand.\n"
                                                                                "several --resnames can be used at once \n if not specified all ligands added with --lig will get analyzed  (unless restart)", required=False)
    parser.add_argument("-geom","--geom", type = str, action="append", default= [], help="""a list of 2-4 of atom selectors eg "resid 131 and name OG1, resname UNK and name N1x" for which you want 
    a a plot of geometric parameters for (distance, angle, dihedral) eg for monitoring near-attack conformations. specify using MDAnalysis/VMD natural language queries""", required=False)
    
    parser.add_argument("-efield_vec","--efield_vec", type = str, action="append", default= [], help="""a list of 3 of atom selectors returning [efield_residue, efield_vec_positive, efield_vec_negative]. 
                        specify using MDAnalysis/VMD natural language queries""", required=False)
    parser.add_argument("-efield_decomp","--efield_decomp", type = str, default= "false", help="Whether to run efield decomposition and pointcharge field calculation", required=False)
    parser.add_argument("-efield_method","--efield_method", type = str, choices=["amber", "amoeba_mutual", "amoeba_extrapolated", "amoeba_direct"], default= "amber", help="What forcefield to use for electric field decomposition", required=False)

    parser.add_argument("-a","--analyze", type = str, default= "True", choices=["true", "True", "false", "False"], help="Run automatic analysis on the resulting trajectory, default True", required=False)
    parser.add_argument("-aca","--center_align", type = str, default= "True", choices=["true", "True", "false", "False"], help="center and align traj. default = True. Set to false when reanalyzing a already center-aligned traj to save time", required=False)
    parser.add_argument("-max_sparse", type=int, default=500, help="How many frames at maximum can the sparse trajectory contain? default=500")

    parser.add_argument("-mtd_cv", "--metadynamics_cv", type = str, action="append", default= [], help="""a pair (bond), triple (angle) or quartet (dihedral) of atom selectors eg "resid 131 and name OG1, resname UNK and name N1x" 
                        which you want to use as colvars for metadynamics. specify using MDAnalysis/VMD natural language queries\n  Dihedrals are assumed to be periodic, eg  E(-180)=E(180) if no aperiodic CV is included.
                        Beware that some CV:s can cause "Particle position is NaN" errors, especially if 3 atoms in a dihedral can become close to linear. If so reconsider your CV choice.
                        Dihedrals go from -180 to 180, Angles from 0 to 180, if wronglly specified no bias will be added to simulation by openmm""", required=False)
    parser.add_argument("-mtd_p", "--metadynamics_parameters", type = str, action="append", default= [], help="A triple of min/max values as well as bias width. In Ångström and degrees." \
    """\nFor bonds, 0.5Å is appropriate whereas 20 degrees is good for angles/dihedrals \nFor example "2, 8, 0.5" """, required = False)
    parser.add_argument("-mtd_bias", "--metadynamics_bias_factor", type = float, default= 4, help="The colvars will be sampled as if they were at temp*bias_factor. default = 4. Higher = we will sample higher energy transitions", required = False)
    parser.add_argument("-mtd_height", "--metadynamics_hill_height", type = float, default= 1, help="The size of gaussian bias which will be added to the simulation at each dumpstep, 1 kJ/mol is default", required = False)
    parser.add_argument("-mtd_bs", "--metadynamics_bias_step", type = int, default= 500, help="How often do we add a gaussian bias to the simulation? 500 steps is default", required = False)
    parser.add_argument("--shift_dihedrals", type = str, default= "False", choices=["true", "True", "false", "False"], help="shifts the discontinuity in a aperiodic torsion from 180 (trans) to 0 (cis). Useful for mixed CV:s", required=False)

    parser.add_argument("--dump_amber", type = str, default= "False", choices=["true", "True", "false", "False"], help="Dump the inpcrd and prmtop for use with amber, consth muste b set to false", required=False)
    parser.add_argument("-ncores", "--ncores", type = int, default= 0, help="How many cores do you want to use for analysis? 0 is default = all availible", required = False)
    parser.add_argument("-plt", "--plot_format", type=str, choices=["png", "svg", "jpg", "tiff"], default="png", help="File format of analyzed plots")
    # Parse arguments
    args = parser.parse_args()

    #print args
    print("Command line:")
    print(f"\n{" ".join(sys.argv[1:])}\n")

    #print command line
    print(f"Parsed arguments: {vars(args)}\n")
    print(parser.description)
    print(parser.epilog + "\n")

    #Sanitize inputs
    if utils.str_to_bool(args.dump_amber) == True and utils.str_to_bool(args.constrain_h) == True and utils.str_to_bool(args.restart) == False:
        raise ValueError("Cannot dump amber files with consth set to True, set consth to False to dump amber files")
    if args.timestep >= 1 and utils.str_to_bool(args.constrain_h) == False and utils.str_to_bool(args.restart) == False:
        raise ValueError("Timestep must be below 1 fs to not constrain hydrogens due to rapid bond vibrations")

    whatcat_md = Whatcat_md_runner(args.pdb, args.lig, utils.str_to_bool(args.restart), args.platform, args.pdbfixer, 
                                   utils.str_to_bool(args.charge_correct), args.solvate, args.ph, utils.str_to_bool(args.constrain_h),
                                   args.timeprod, args.timestep, args.report_time, args.equillibration_time, args.plot_format,
                                   utils.str_to_bool(args.pbc))

    if whatcat_md.restart is not True:
        whatcat_md.fix_pdb()

        #Handle custom variants if provided
        if len(args.variants) == 0:
            variants = None
        else:
            variants = {}
            for variant in args.variants:
                res_idx, resname = utils.css_to_list(variant)
                #-1 to convert from 1 indexing to 0 indexing
                variants[int(res_idx)-1] = resname

        whatcat_md.create_openmm_system(variants = variants)
        whatcat_md.create_openmm_simulation()
        whatcat_md.equillibrate_simulation(minimize=utils.str_to_bool(args.minimize))
    elif whatcat_md.restart is True:
        whatcat_md.restart_simulation_from_file()

    if len(args.metadynamics_cv) == 0:
        whatcat_md.run_prod_simulation()

    elif len(args.metadynamics_cv) > 0:
        #unpack the list of comma separated selection strings to a list of lists with atom indicies in them
        atom_indices_list = [[utils.atom_idx_from_selection(selection, whatcat_md.simulation.topology) for selection in utils.css_to_list(cv)] for cv in args.metadynamics_cv]

        #read in the user input for parameters
        colvar_parameters = []
        for param in args.metadynamics_parameters:
            parameter_list = utils.css_to_list(param)
            colvar_parameters.append([float(x) for x in parameter_list])

        whatcat_md.add_metadynamics(atom_indices=atom_indices_list, colvar_parameters=colvar_parameters, bias_factor=4, hill_height = 1, shift_dihedrals=utils.str_to_bool(args.shift_dihedrals))
        pes = whatcat_md.run_metadynamics_simulation()

    #dump amber files if requested
    if utils.str_to_bool(args.dump_amber) == True:
        whatcat_md.dump_amber_files()
        
    if utils.str_to_bool(args.analyze):
        #set default for analysis
        if len(args.geom) != 0:
            analysis_distances = args.geom
        elif len(args.metadynamics_cv) != 0:
            analysis_distances = args.metadynamics_cv
        else:
            analysis_distances = []

        analysis_resnames = utils.prepend_list(args.resname, "resname ")

        #if data is availible set that
        if len(args.resname) == 0:
            analysis_resnames = whatcat_md.analysis_resnames
        
        whatcat_analysis = whatcat_md.create_analysis(plot = True, max_sparse_frames=args.max_sparse, ncores = args.ncores)

        whatcat_analysis.read_md_log()
        whatcat_analysis.equillibration_check()

        if utils.str_to_bool(args.center_align):
            whatcat_analysis.center_align_traj() 

        whatcat_analysis.calc_rgyr()
        whatcat_analysis.calc_1d_rmsd(analysis_resnames + ["backbone"])
        whatcat_analysis.calc_ca_rmsf()

        if len(analysis_distances) > 0:
            whatcat_analysis.calc_geometric_params(analysis_distances)

        whatcat_analysis.write_sparse_traj()
        whatcat_analysis.calc_2d_rmsd(analysis_resnames + ["backbone"])

        if len(args.efield_vec) > 0:
            for efield_sel in args.efield_vec:
                sel_list = utils.css_to_list(efield_sel)
                
                if len(sel_list) != 3:
                    raise ValueError("bad input.\n-efield_vec did not contain three selections. \nRevise input")

                whatcat_analysis.calc_efield(sel_list[0], sel_list[1:3], 
                                             run_decomp=utils.str_to_bool(args.efield_decomp), 
                                             run_pointcharge=utils.str_to_bool(args.efield_decomp),
                                             method=args.efield_method)

        #separate call to clustering for backbone since we actually care about the data (To plot SPM structure)
        _, backbone_cluster_df= whatcat_analysis.cluster_trajectory(selection="backbone", method_name="hdbscan")

        if len(whatcat_analysis.u.trajectory) > 5:
            for selection in analysis_resnames:
                whatcat_analysis.cluster_trajectory(selection=selection, method_name="hdbscan")
            
            for selection in analysis_resnames + ["backbone"]:
                whatcat_analysis.dimensionality_reduction(method_name="pca", selection=selection, n_components=3)
        else:
            print("To few snapshots for clustering and PCA")

        whatcat_analysis.remove_sparse_traj()
        
        #Write most populated backbone cluster for distance matrix calc
        spm_pdb = f"{whatcat_analysis.basename}_SPM.pdb"
        whatcat_analysis.u.trajectory[backbone_cluster_df["Centroid_frame"].iloc[0]-1] #-1 to convert from 1 indexing to 0 indexing
        whatcat_analysis.u.select_atoms("not resname HOH and not resname CL and not resname NA").write(spm_pdb) 
        whatcat_analysis.calc_distance_matrix(structure=spm_pdb)

        whatcat_analysis.calc_correlation_matrix()

        if len(analysis_resnames) > 0:
            whatcat_analysis.run_prolif(analysis_resnames = analysis_resnames)

        whatcat_analysis.save_df_to_csv()

        #print(whatcat_analysis.time_df.head())
        #print(whatcat_analysis.residue_df.head())

if __name__ == "__main__":
    main()

