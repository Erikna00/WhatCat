from openmm.app import *
from openmm import *
from openmm.unit import *
import sys
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer
import argparse
import numpy as np
from utils import utils
#from utils import analysis
import re
import warnings
# suppress some MDAnalysis warnings when writing PDB files as well as the DCD timestep warning
warnings.filterwarnings('ignore')
#filter biopython warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="Bio.Application")


#TODO improve metalloprotein handling https://ash.readthedocs.io/en/latest/Metalloprotein-I.html
#TODO add the ability to run sequential replicates of the same simulation via argparse
#TODO add analysis to Whatcat_md or make it an own class
#TODO Fix RMSF bug
#TODO consider redoing the class  such that all arguments are provided only to the function when they are used instead of __init__
#then it would make sens to move argparse to if __name__ = main

class Whatcat_md():
    def __init__(self, 
                 pdb_file, ligand_files = None, restart = False,  platform="CUDA",
                 pdb_fixer=2, charge_correct = True, solvate = 2, ph = 7.4,
                 simulation_time_ns=None, timestep=4, reporting_time=1, equillibration_time=50,
                 analysis_resnames =[], analysis_distances = [], 
                 debug = False ):
        """
        Creates a Whatcat_md object from python arguments.
        A Whatcat_md object can also be created via init_from_parse_args()

        pdb_file - path to pdbfile, must not contain any small molecules
        ligand - path to ligand file, must be sdf if charge_correct = False, charge_correct converts with openbabel
        restart - bool for if to restart from _restart.xml files printed by a previous run. Also appends to existing reporter path
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
        
        analysis_resnames - for which MDAnalysis selctions to run more specific analysis
        analysis_distances - which distances shall be monitored during the simulation
        """
        #TODO should __init__ have defaults? currentlly we can initillize values from all functions anyway
        
        #Extract into self.varibles
        self.pdb_file = pdb_file
        self.ligand_files = list(ligand_files)
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

        self.analysis_resnames = list(analysis_resnames)
        self.analysis_distances = list(analysis_distances)
        
        self.debug = debug

        self.pdb_name = os.path.splitext(pdb_file)[0]
        self.ran_time = 0 *picoseconds

        try:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        except:
            self.script_dir = os.getcwd()


        #Restart overrides settings to allow loading of pdb_final
        #TODO is this reasonable to do here after code refactoring to class?
        if restart == True:
            self.pdb_name = self.pdb_name.replace("_restart", "")

    @classmethod
    def init_from_parse_args(cls):
        """
        Starts a Whatcat_md class using command line arguments to run __init__
        """

        #Start the command line parser
        parser = argparse.ArgumentParser(
                            prog='POS MD script',
                            description=(
                            "This script sets up a OPENMM  simulation of a protein (amber ff14)," 
                            "any small molecules/cofactors (Sage 2.2.1) as well as water/ions using a 12-6 model (amber ff14/tip3pfb).\n"),
                            epilog="Use with care and acknowledge Erik Sundén and the Per-Olof Syrén group at KTH Sweden")

        parser.add_argument("pdb", type = str, help = "PDB structure of the structure you want to simulate. \nWARNING PDB may not contain any ligands. These must be provided from sdf files") 
        parser.add_argument("-l", "--lig", type = str, action="append", default = None, help = ("optional parameter, SDF file containing all nonstandard ligands and cofactors." 
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
        parser.add_argument("-rt", "--report_time", type = float, default= 1, help="Reporting frequency in ps")
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

        return cls(pdb_file=args.pdb, ligand_files = args.lig, restart = utils.str_to_bool(args.restart), platform=args.platform, 
                 pdb_fixer=args.pdbfixer, charge_correct = args.charge_correct, solvate = args.solvate, ph = args.ph,
                 simulation_time_ns=args.timeprod, timestep=args.timestep, reporting_time=args.report_time, equillibration_time=args.equillibration_time,
                 analysis_resnames =args.resname, analysis_distances = args.dist, 
                 debug = args.debug)

    def parse_default(self, attr_name, value):
        """
        Parses method variables compared to class variables to allow
        methods to use both class and set variables
        valid call is self.parse_default("charge_correct", charge_correct)
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
        pdb_fixer = self.parse_default("pdb_fixer", pdb_fixer) 

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
        pdb = self.parse_default("pdb", pdb)
        charge_correct = self.parse_default("charge_correct", charge_correct)
        ph = self.parse_default("ph", ph)
        solvate = self.parse_default("solvate", solvate)
        
        #TODO add accellerated MD without colvars or metadynamics with colvars 
        #forcefield kwargs
        forcefield_kwargs = {'constraints': HBonds, 'rigidWater': True, 'removeCMMotion': True, 'hydrogenMass' : 1.5 * amu }

        #if simulating with ligand
        if self.ligand_files is not None:
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

            #if analysis not specified, analyze all added residues
            if self.analysis_resnames is None:
                self.analysis_resnames = lig_resnames

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

            #if no analysis requested add all ligands
            if len(self.analysis_resnames) == 0:
                for ligand_name in lig_resnames:
                    self.analysis_resnames.append(ligand_name)
            
        #if not simulating with ligand
        elif self.ligand_files == None:
            # Specify the forcefield
            # Initialize a SystemGenerator using the Sage.2.1 for the ligand and tip3p for the water.
            
            system_generator = SystemGenerator(
                forcefields=['amber14-all.xml', 'amber14/tip3pfb.xml'],
                small_molecule_forcefield='openff-2.2.1.offxml',
                forcefield_kwargs=forcefield_kwargs, cache=cache_file)
            
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
        system = self.parse_default("system", system)
        modeller = self.parse_default("modeller", modeller)
        timestep = self.parse_default("timestep", timestep)

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
        Requieres that pdb_file is set to _final.pdb from a previous simulation
        Alternativelly pdb_file can be set to a path to the _restart.pdb file.
        If pdb_file is set, self.pdbfile is overwritten and self.name is set to the restart_pdb_file with _restart removed
        else self.pdb_name is inspected and _restart is removed if present
        """

        if restart_pdb_file is not None:
            self.pdb_file = restart_pdb_file
            #remove restart if present
            self.pdb_name = restart_pdb_file.replace("_restart", "")
        
        self.pdb_name = self.pdb_name.replace("_restart", "")
        
        #set file basename of all restart files
        checkpoint_filebase = f"{self.pdb_name}_restart"

        pdb = PDBFile(f"{checkpoint_filebase}.pdb")
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
        print(f"\nSimulation restarted with stepsize {self.timestep}\n")

        self.simulation = simulation
        self.ran_time = simulation.context.getState().getTime()

        return simulation
    
    def equillibrate_simulation(self, simulation = None, equillibration_time=None):
        """
        Takes a openmm simulation object and equillibrates it for the provided time converting it
        to a NPT simulation in the process
        """

        #TODO maybe separate NVT and NPT equillibration time?
        simulation = self.parse_default("simulation", simulation)
        equillibration_time = self.parse_default("equillibration_time", equillibration_time)

        equillibration_steps = int(equillibration_time / (self.timestep * 10**-3))

        print("Minimizing energy")
        simulation.minimizeEnergy()

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

        self.simulation = simulation
        self.equillbration_steps = equillibration_steps
        self.ran_time += 2*equillibration_time *picoseconds

        return simulation

    def run_prod_simulation(self, simulation = None, simulation_time_ns=None, reporting_time = None):
        """
        Runs the production NPT simulation for the set amount of time.
        Re
        """
        simulation = self.parse_default("simulation", simulation)
        simulation_time_ns = self.parse_default("simulation_time_ns", simulation_time_ns)
        reporting_time = self.parse_default("reporting_time", reporting_time)

        #calculate simulation length
        production_steps = int(simulation_time_ns / (self.timestep * 10**-6))
        reporting_frequency = int(reporting_time / (self.timestep * 10**-3))

        #add reporters
        #print to terminal
        totalsteps = int(self.ran_time.value_in_unit(picoseconds)) * 1000/self.timestep + production_steps
        simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True,
                potentialEnergy=True, temperature=True, volume=True, remainingTime=True, totalSteps= totalsteps, speed=True))

        #saved to file
        simulation.reporters.append(StateDataReporter(f"{self.pdb_name}_md_log.txt", reporting_frequency, step=True,
                potentialEnergy=True, temperature=True, volume=True))
        simulation.reporters.append(DCDReporter(f"{self.pdb_name}_trajectory.dcd", reporting_frequency, append = self.restart))

        print("Running production NPT")
        simulation.step(production_steps)

        #save pdb
        state = simulation.context.getState(getPositions=True)
        with open(f"{self.pdb_name}_final.pdb", "w") as file:
            PDBFile.writeFile(simulation.topology, state.getPositions(), file)
        with open(f"{self.pdb_name}_restart.pdb", "w") as file:
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

#TODO create a wrapper function of this that presents the user with a equillibrated simulation
#right away
whatcat_md = Whatcat_md.init_from_parse_args()
if whatcat_md.restart is not True:
    whatcat_md.fix_pdb()
    whatcat_md.create_openmm_system()
    whatcat_md.create_openmm_simulation()
    whatcat_md.equillibrate_simulation()
elif whatcat_md.restart is True:
    whatcat_md.restart_simulation_from_file()

simulation = whatcat_md.run_prod_simulation()

simulation_time_ns = whatcat_md.simulation_time_ns
reporting_time = whatcat_md.reporting_time
pdb_name = whatcat_md.pdb_name
analysis_resnames = whatcat_md.analysis_resnames
analysis_distances = whatcat_md.analysis_distances
debug = whatcat_md.debug

