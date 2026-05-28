from openmm.app import *
from openmm import *
import openmm.unit as unit
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator

import MDAnalysis as mda

from sys import stdout
import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke
import matplotlib as plt
import argparse
import time
from tqdm.auto import tqdm

import whatcat.md.utils.utils as utils


class _Efield_base():

    def __init__(self, topology, trajectory, scan_resid_sel, efield_vec_sel,  platform="CUDA", 
                 precision = "single", solvent_names=("HOH", "CL", "NA"), 
                 extra_decomp_residue_sel=["name MG", "name ZN"], sparsity=1):
        """
        Takes a finished simulation and does GPU accellerated electrostatic field calculations.
        This is still quite slow compared to normal MD if a polarizable force field is used.

        Internally this class uses OpenMM indexing which is 0 indexed for both atoms and residues.
        Usage with Mdanalysis uses normal MDA indexing which is 0 indexed for atoms and 1 indexed for residues.
        Hence one should never change internal variables after class creation to avoid confusion.

        If the scanned residue is part of the protein the code does run but it does not remove the scanned
        residues contribution to the field which will be quite large

        This class works with electric fields in kJ/(mol*debye)
        To convert to the experimental unit MV/cm multiply by 4.978170852464958

        For a reference workload the relative runtimes traj R^2 and decomp R^2 (amoeba mutual as reference) were
            Efield_amber                            3%, 0.60, 0.72
            Efield_amoeba polarization Direct       59%, 0.74, 0.28
            Efield_amoeba polarization Extrapolated 79%, 0.996, 0.999
            Efield_amoeba polarization Mutual       100%, 1, 1

        Parameters:
            topology : str or OpenMM.Topology
                Path to pdb file or OpenMM.Topology object.
            trajectory : str
                Path to trajectory file, e.g. dcd.
            scan_resid_sel : str
                MDAnalysis selection string for the residue in which you want the efield calculation
                to happen.
            efield_vec_sel : tuple of str
                2 member tuple of selections yielding 1 or more atoms each who define the approximate 
                transition state dipole moment. The fictional particle will be added halfway between these.
                By convention the tuple should be (positive charge, negative charge) in the TS.
                This means all stabilizing fields will have a negative sign.
            platform : str
                The OPENMM platform you want to run this class in.
                Valid choices are "CUDA", "HIP", "OpenCL" or "CPU"
                case sensitive
            precision : str
                What precision to evaluate forces on.
                valid choices are single, double
                Choosing single gives a 15x speedup but leads to a small random
                error on the fifth decimal and below
            solvent_names : tuple of str
                The residue names which are characterized as solvent.
            extra_decomp_residue_sel : list of mdanalysis selection strings
                Extra selections for residues/atoms which should enter the decomposition.
                These residues are not included in the get_pointcharge_field() method.
            sparsity : int
                How often do you want to analyze the trajectory? 1=every frame

        """
        #Notes
        #In MDAnalysis resids are 1 indexed and atoms are 0 indexed
        #In OPENMM both are 0 indexed
        #This is evidentlly because god hates us
        #We will standardize on OPENMM indexing 

        #Openmm works in nm MDAnalysis in Angstrom
        #We will work in Ångström 

        #Unit conversions
        
        #Convert to MV/cm which is the experimental unit in stark spectroscopy
        # kJ/(mol*nm) to MV/cm, F/q = E
        #10^9 nm > m
        #1000/avogadro = kj/(mol*m) to J/m = Newton
        #/elementary charge gives electric field in V/m

        # /(100*10^6) to MV/cm
        #AVOGADRO = 6.0221408e23
        #ELEMENTARY_CHARGE = 1.60217663e-19
        #LIGHTSPEED = 299792458
        #conversion_factor = (1e9 * 1000) / (AVOGADRO * ELEMENTARY_CHARGE * 1e8) = 0.10364269613296549

        #Similairly for kJ/(mol*nm) to kJ/(MOL*DEBYE)
        # E = F *dipole 
        # (1e9 * 1000) / (AVOGADRO * ELEMENTARY_CHARGE) gives V/M from kJ/(mol*nm)
        # 1/unit.SPEED_OF_LIGHT_C * 1e-21 #debye to C*M
        # C*M * V/M = J
        # J/molecule = (1e9 * 1000 * unit.SPEED_OF_LIGHT_C * 1e+21) / (AVOGADRO * ELEMENTARY_CHARGE)
        # *AVOGADRO/1000 to kj/mol
        #conversion_factor = (1e9 * 1000* 1e-21 * AVOGADRO) / (AVOGADRO * ELEMENTARY_CHARGE* LIGHTSPEED * 1000) = 0.02081943332291347

        #Topology and system related stuff
        if type(topology) == str:
            #Read pdb file into openmm topology
            self.topology = PDBFile(topology).getTopology()
        else:
            #Hopefully an openmm topology object
            self.topology = topology

        self.trajectory = trajectory
        self.u = mda.Universe(topology, trajectory) #The universe is rebuilt during setup
        self.scan_resid_sel = scan_resid_sel
        self.extra_decomp_residue_sel = extra_decomp_residue_sel

        #Handle sparsity
        if sparsity != 1:
            sparse_traj = "sparse_trajectory.dcd"
            with mda.Writer(sparse_traj, n_atoms=self.u.atoms.n_atoms) as writer:
                for ts in self.u.trajectory[::sparsity]:
                    writer.write(self.u.atoms)
                self.trajectory = sparse_traj
                self.u = mda.Universe(self.topology, sparse_traj)

        #Find script dir where the ligand cache is stored
        try:
            self.script_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"
        except:
            self.script_dir = os.getcwd()

        #Empty variables to be filled later

        self.system = None
        self.modeller = None
        self.context = None
        self.simulation = None
        self.integrator = None

        #Not used but needed for openmm simulation creation
        self.timestep = 1 * unit.femtoseconds
        self.temperature = 300 * unit.kelvin
        self._platform = platform
        self._precision = precision

        self.solvent_names = solvent_names

        #Stored as np array of (atom_index, residue_index)
        self._protein_and_interesting_array = None

        #stored as np array of atom_index
        self._solvent_array = None

        #Position related variables
        self.efield_vec_sel = efield_vec_sel
        self.efield_vec_direction = None #assigned at particle creation (_get_fictional_particle_positions()), (frame, coord) array of length 1 vectors
        self.fictional_particle_index = None #assigned at particle creation (_get_fictional_particle_positions())
        self.fictional_particle_pos = None #assigned at particle creation (_get_fictional_particle_positions())

        #Run some internal setup specific to subclass
        self._run_setup()

    def _run_setup(self):
        """
        Runs internal setup specific to a certain subclass.
        """
        raise NotImplementedError("Not implemented, this is a baseclass")
    
    def _add_fictional_particle_parameters(self):
        """ 
        Adds the fictional particle parameters to the system.
        Needs to be implemented for each force field as the nonbonded handling differs between force fields.

        Parameters:
            None

        """
        raise NotImplementedError("Not implemented, this is a baseclass")

    def _init_params(self):
        """
        Initialize parameters and modify NonbondedForce for energy decomposition.
        Run in create_system()

        Forcegroups
        0 = not intresting
        1 = nonbonded
        """
        raise NotImplementedError("Not implemented, this is a baseclass")
    
    def _get_fictional_particle_positions(self):
        """ 
        This method gets the position where the fictional particle should be at
        in each frame of the trajectory and then writes a new trajectory with the position
        added in.

        It also revises the trajectory after the new topology to accomodate the
        deletion of the scanned residue and other system changes as well as saves the direction
        of the efield vector on each frame.

        Parameters
            None
        
        Returns: 
            np array of shape [frame, 3] where each entry is a x,y,z coordinate in Å
        """
        #at this point self.u is already revised to not have ligand and has the fictional particle added
        #create a new original universe to read the original DCD
        u_original = mda.Universe(self.topology, self.trajectory)

        #Explicitlly make sure we are at first frame of traj
        u_original.trajectory[0]
        ags = []

        #Retrieve particle information from sel flagging if we have a multiatomic case
        for sel_str in self.efield_vec_sel:
            ag = u_original.select_atoms(sel_str)

            #Sanitize user input and give a heads up for mistakes
            if len(ag) == 0:
                raise ValueError(f"No atoms inside {sel_str}")
            elif len(ag) > 1:
                print(f"There is {len(ag)} atoms selected by {sel_str}, hope this was intentional")
            ags.append(ag)

        #select the ligand for deletion later
        ligand_ix = u_original.select_atoms(self.scan_resid_sel).ix_array

        #Construct array from the trajectory
        fictional_particle_pos = []
        efield_vec_direction = []
        traj_name = self.trajectory.split(".")[0] + "_dummy.dcd"

        with mda.coordinates.DCD.DCDWriter(traj_name, n_atoms=self.u.atoms.n_atoms) as w:
            for ts in u_original.trajectory:
                #Calc atomgroup midpoints
                ag0_mid = np.average(ags[0].positions, axis=0)
                ag1_mid = np.average(ags[1].positions, axis=0)

                #Calc center between atomgroups to put in the fictional particle there
                particle_position = np.average([ag0_mid, ag1_mid],axis=0)

                #remove scan residue from traj
                traj = ts.positions
                traj = np.delete(traj, ligand_ix, axis=0)

                #Append fictional particle to output array
                fictional_particle_pos.append(particle_position)

                # Append the extra particle's position to the current frame
                new_traj_positions = np.vstack([traj, particle_position])
                self.u.trajectory.ts.positions = new_traj_positions

                #Add periodic box vectors from original traj
                self.u.trajectory.ts.dimensions = ts.dimensions
                w.write(self.u.trajectory)

                #Also calculate the efield vector of this frame
                vec = ag1_mid-ag0_mid #Get ag0 > ag1 vector
                vec = vec / np.linalg.norm(vec) #divide by vector length to get length 1 vector
                efield_vec_direction.append(vec)

        self.trajectory = traj_name
        self.efield_vec_direction = np.array(efield_vec_direction)
        self.fictional_particle_pos = np.array(fictional_particle_pos)

        return np.array(fictional_particle_pos)

    def _get_atomgroups(self):
        """
        Constructs all index-residue arrays for later use
        
        Returns:
            None
        """
        #Make solvent array of atom indices
        solvent = self.u.select_atoms(" or ".join([f"resname {sel}" for sel in self.solvent_names]))
        self._solvent_array = solvent.ix_array

        #setup selection string with eventual extra residues/ions
        if len(self.extra_decomp_residue_sel) == 0:
            selection_str = "protein"
        else:
            selection_str = "protein or " + " or ".join(self.extra_decomp_residue_sel)

        #Make protein array of (atom_index, residue_index)
        protein = self.u.select_atoms(selection_str)
        self._protein_and_interesting_array = np.empty([len(protein.atoms), 2], dtype=int)

        self._protein_and_interesting_array[:,0] = protein.atoms.ix_array
        self._protein_and_interesting_array[:,1] = protein.atoms.resids
        self._protein_and_interesting_array[:,1] -= 1 #Convert residues to 0 indexed

    def _create_system(self):
        """
        Creates the OPENMM topology which will be used for the interaction energy calculation.
        This function is a derivative of whatcat.md.md.Whatcat_md_runner.create_system.
        This function does not do charge correction of ligand based on pH so be sure your sdf is more
        sanitized than when running normal MD.

        This function also adds a +1 charged fictional particle to the system which we use to measure the electric
        field.

        This function is ran automatically during init()
        
        Parameters:
            None

        """

        #forcefield kwargs
        forcefield_kwargs = {"constraints": HBonds, "rigidWater": True, "removeCMMotion": True, "hydrogenMass" : 1.5 * unit.amu }


        cache_file = f"{self.script_dir}/ligands.json"
        lig_namer = utils.Ligand_namer()
        ligand_mol = []
        lig_resnames = []

        #start a cache finder object
        cache_finder = utils.FF_cache_reader(cache_file=cache_file)

        # Specify the forcefield
        # Initialize a SystemGenerator using the Sage.2.1 for the ligand and tip3p for the water.
        system_generator = SystemGenerator(
            forcefields=self._biomolecule_ff,
            forcefield_kwargs=forcefield_kwargs)
        
        #start a modeller
        #Convert from MDAnalysis Angstrom to openmm nm
        modeller = Modeller(self.topology, self.u.trajectory[0].positions * unit.angstroms) 

        #delete scan residue in modeller if it is not protein (handled by scan_resid_sel selection string)
        selection_atom_ix = self.u.select_atoms(self.scan_resid_sel).ix_array
        atoms = list(modeller.topology.atoms())
        atoms_to_del = [atoms[i] for i in selection_atom_ix]
        modeller.delete(atoms_to_del)

        # Create the system using the SystemGenerator
        system = system_generator.create_system(modeller.topology)

        #set precision and platform
        platform_name, device_idx = utils.platform_device_sanitizer(self._platform)

        if platform_name == "OpenCL" or platform_name == "CUDA" or platform_name == "HIP":
            properties = {"Precision":self._precision, "DeviceIndex":device_idx} #improves energy conservation resulting in larger stable timesteps. decreases speed ca 10%
            platform = Platform.getPlatformByName(platform_name)
        elif platform_name == "CPU":
            properties = {}
            platform = Platform.getPlatformByName(platform_name)

        #Here we add the fictional particle to the system and modeller
        #We add it to its own chain and residue since openmm requieres residues/chains to be
        #Contignouslly bonded
        chain = modeller.topology.addChain()
        res   = modeller.topology.addResidue("DUM", chain)
        atom = modeller.topology.addAtom("DUM", Element.getBySymbol("He"), res, formalCharge=1)
        self.fictional_particle_index = atom.index #save index for later
        #finctional_particle is appended as the last atom in the topology

        # Add a fake position from frame 0 
        modeller.positions = np.append(modeller.positions.value_in_unit(unit.angstrom), [[0,0,0]], axis=0) * unit.angstrom 

        #Create atomgroup of only fictional particle and merge into universe
        fictional_particle = mda.Universe(modeller.topology, modeller.positions, format=mda.coordinates.memory.MemoryReader).atoms[-1:] 
        pruned_topo = self.u.select_atoms(f"not ({self.scan_resid_sel})") #select everything we did not remove in original traj
        
        #make universe from merging atomgroups
        #This solves the immutable universe problem
        self.u = mda.Merge(pruned_topo, fictional_particle) 
        
        #Get the positions of the fictional particle and revise trajectory after new topology
        particle_position = self._get_fictional_particle_positions()[0,:]

        #place fictional particle correctlly in modeller
        modeller.positions[-1] = particle_position * unit.angstrom 

        self.modeller = modeller
        self.system = system

        #Add the fictional particle to system along with parameters
        self._add_fictional_particle_parameters()

        #We need to initiallize the context parameters and forcegroups before making the simulation
        #For this we need to start the atomgroup arrays
        self._get_atomgroups()
        self._init_params()

        #Now we can finally build the simulation and context
        self.integrator = LangevinMiddleIntegrator(self.temperature, 1/unit.picosecond, self.timestep)
        self.simulation = Simulation(modeller.topology, system, self.integrator, platform = platform, platformProperties=properties)
        self.simulation.context.setPositions(modeller.positions)
        self.context = self.simulation.context

        #Reassign universe so it tracks the simulation topology and trajectory with the fictional particle
        self.u = mda.Universe(self.simulation.topology, self.trajectory)

    def _get_residue_positions(self):
        """
        Fetch the residue positions (relative to fictional particle).
        Linear algebra is used to compute a virtual Cb position from C, Ca and N.
        This allows homogenous treatment of all amino acids including glycine and proline
        This is a utils function for self.get_pointcharge_field()

        Returns:
            np.ndarray : array containing fictional particle to residue vectors
            over the trajectory of shape [frame, residue, vector]

        """
        #make a atomgroup of all intresting residues
        Ca_ag = self.u.residues[np.unique(self._protein_and_interesting_array[:,1])].atoms.select_atoms("protein and name CA")
        C_ag = self.u.residues[np.unique(self._protein_and_interesting_array[:,1])].atoms.select_atoms("protein and name C")
        N_ag = self.u.residues[np.unique(self._protein_and_interesting_array[:,1])].atoms.select_atoms("protein and name N") 

        fictional_ag = self.u.select_atoms("name DUM")

        #Preallocate a array for generated data
        residue_positions = np.empty([len(self.u.trajectory), len(Ca_ag), 3], dtype=np.float32)

        #define the Cb internal coordinates (fitted on SvS-A2 MD traj, excluding gly and pro, RMSD 0.13Å)
        dih_C_N_Ca_Cb = np.deg2rad(129.4) #degrees, 2/3 from tetrahedral angle to water angle
        Ca_Cb_bondlength = 1.535 #Å

        for index, ts in enumerate(self.u.trajectory):
            #retrieve all atom positions
            Ca_pos = Ca_ag.positions
            C_pos = C_ag.positions
            N_pos = N_ag.positions

            #make vectors relative to residue CA
            C_pos = C_pos - Ca_pos
            N_pos = N_pos - Ca_pos

            #Normalize vector lengths to 1
            C_pos /= np.linalg.norm(C_pos, axis=1, keepdims=True)
            N_pos /= np.linalg.norm(N_pos, axis=1, keepdims=True)

            #calculate the normalized bisector vector (in Ca-C-N plane basis vector)
            C_N_bisect = C_pos + N_pos
            C_N_bisect = C_N_bisect / np.linalg.norm(C_N_bisect, axis=1, keepdims=True)

            #Calculate and normallize cross product (basis vector perpendicular to Ca-C-N plane)
            NxC = np.cross(N_pos, C_pos, axis=1)
            NxC /= np.linalg.norm(NxC, axis=1, keepdims=True)

            #use a rotation in the ortogonal basis of the Ca-Ha-Cb plane to compute Cb vector
            Cb_pos = NxC * np.sin(dih_C_N_Ca_Cb) + C_N_bisect * np.cos(dih_C_N_Ca_Cb)
            Cb_pos /= np.linalg.norm(Cb_pos, axis=1, keepdims=1)/Ca_Cb_bondlength

            #convert back to the absolute reference frame
            Cb_pos += Ca_pos

            #convert to fictional particle reference frame
            Cb_pos -= fictional_ag.positions

            #Write calculated positions to output
            residue_positions[index] = Cb_pos

        return residue_positions


    def get_pointcharge_field(self, raw=0):
        """
        Runs a default workflow for calculating the resulting electric field component from introducing a charge at the Cb
        position of every residue of the protein. For consistency the Cb position is calculated from backbone position to handle
        proline and glycine consistentlly with the rest.
        This can be used to guide mutagenesis to influential residues.

        The returned values are in kJ/(mol*D) and a negative value means that introducing a positive
        charge at Cb would exert a field in the opposite direction to your defined efield_vec.
        If you did this as requested, positive > negative, then a negative field stabilizes TS.

        Parameters:
            raw: bool
                Regulates what is returned from the function
                False= project the force on the reference vector and average over trajectory
                True= return the raw force vector frame by frame

        Returns:
            np.ndarray: exerted electric field by a positive charge introduced
                at Cb in kJ/(mol*D).
                raw = False shape (residues)
                raw = True shape (frames, residues)
        """
        #The coloumb constant is derived as
        #ke = 8.98755*10**9 N*m**2/C**2
        #ke = ke * 1.602176 * 10**-19 N*m**2/e**2
        #Newton = J/m
        #ke = ke * 6.022 141 *10**23 / 1000 kJ*m/(mol*e**2)
        #ke = ke * 10**10 kJ*Å/(mol*e**2)
        # 1 debye = 0.2081943 e*Å
        # ke = ke * 0.2081943 kJ*Å**2/(mol*e*D)
        # Efield = ke * q/r**2
        coloumb_constant = 289.2554 #kJ*Å^2/(mol*e*D)
        
        #get residue positions relative to fictional particle
        residue_positions_all = self._get_residue_positions()

        #Start the results array
        result = np.empty(residue_positions_all.shape[0:2], dtype=np.float32)
        
        #iterate over trajectory frames by slicing along first axis
        for index, residue_positions_frame in enumerate(residue_positions_all):
            #calculate field size
            distances = np.linalg.norm(residue_positions_frame, axis=1, keepdims=True)
            fields = -coloumb_constant /distances**2

            #Make vector of the force
            fields_vec = residue_positions_frame * (fields / distances)

            #Project onto reference vector
            result[index,:] = np.dot(fields_vec, self.efield_vec_direction[index])

        #average over trajectory frames
        if raw == False:
            result = np.average(result, axis=0)

        return result
    
    def delete_traj(self):
        """
        Delete the trajectory containing the dummy particle.
        """

        os.remove(self.trajectory)
        self.trajectory = None
    
class Efield_amber(_Efield_base):

    def _run_setup(self):
        """
        Runs internal setup specific to amber ff
        """

        self._biomolecule_ff = ["amber14-all.xml", "amber14/tip3pfb.xml"]
        self._create_system()
        

    def _init_params(self):
        """
        Initialize parameters and modify NonbondedForce for energy decomposition.
        Run in create_system()

        Forcegroups
        0 = not intresting
        1 = NonbondedForce
        """
        #Iterate over all forces and divide into forcegroups

        for force in self.system.getForces():
            #Find the one NB force
            if isinstance(force, NonbondedForce):
                #Assign nonbonded forcegroup and add the solvent scale
                force.setForceGroup(1)

                #Add parameter, default on
                force.addGlobalParameter("solvent_coul_scale", 1)

                #Create per-residue force parameter
                for i in np.unique(self._protein_and_interesting_array[:,1]):
                    #Add parameter, default on
                    force.addGlobalParameter(f"residue{i}_coul_scale", 1)
                
                #Iterate over atoms and revise forces for decomposition
                for atom_index in range(force.getNumParticles()):
                    #Get original parameters
                    charge, sigma, epsilon = force.getParticleParameters(atom_index)

                    # Set default interaction to zero
                    force.setParticleParameters(atom_index, 0, 0, 0)

                    #Check if atom is protein
                    if atom_index in self._protein_and_interesting_array[:,0]:
                        #Setup per-residue parameter
                        #Get the residue number by first finding row of atom and then checking second column
                        residue_number = self._protein_and_interesting_array[np.where(self._protein_and_interesting_array[:,0] == atom_index)[0], 1][0]
                        force.addParticleParameterOffset(f"residue{residue_number}_coul_scale", atom_index, charge, 0, 0)

                    #If atom is solvent, no need to slice as we only store atom indexes
                    elif atom_index in self._solvent_array:
                        #Set into solvent nonbonded forcegroup
                        force.addParticleParameterOffset("solvent_coul_scale", atom_index, charge, 0, 0)
                    
                    #Add back original parameters to the fictional particle since it is perfect
                    elif atom_index == self.fictional_particle_index:
                        force.setParticleParameters(atom_index, charge, sigma, epsilon)
                    
                    #If any particle has not been handled something is very wrong
                    else:
                        raise ValueError(f"uncaught particle of index {atom_index}, make a bug report")

                #This handles Ambers 1-4 parameters by setting to zero
                #Since this is a bonded dihedral interaction it will not affect our nonbonded decomposition
                #In any case the fictional particle is not bonded to anything so it is unaffected
                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                    force.setExceptionParameters(i, p1, p2, 0, 0, 0)
         
            elif isinstance(force, CustomNonbondedForce):
                raise ValueError("System contains custom nonbonded forces, did you hack in a CHARMM ff?\n" \
                "This is not allowed for the amber efield calculator")

            else:
                #Set not nonbonded forces to force group 0
                force.setForceGroup(0)
    
    def reset_context(self, prot_coul=1, solv_coul=1):
        """
        Reset context parameters to defaults.
        Coloumb interactions are set to a value of 1 aka all forces are on except 
        the ones pertaining to the residue the fictional particle is inside which are 0.

        Parameters:
            prot_coul : float
                Scaling factor for protein coulombic parameters (default 1)
            solv_coul : float
                Scaling factor for solvent coulombic parameters (default 1)

        Returns:
            None
        """
        for i in np.unique(self._protein_and_interesting_array[:,1]):
            self.context.setParameter(f"residue{i}_coul_scale", prot_coul)

        self.context.setParameter(f"solvent_coul_scale", solv_coul)
    
    def _add_fictional_particle_parameters(self):
        """ 
        Adds the fictional particle parameters to the system.
        Needs to be implemented for each force field as the nonbonded handling differs between force fields.

        Parameters:
            None

        """
        #Add the fictional massless particle to the system 
        fictional_particle_index = self.system.addParticle(0.0)  

        #Find nonbonded force
        for f in self.system.getForces():
            if isinstance(f, NonbondedForce):
                test = f.addParticle(charge=1.0, sigma=0.0, epsilon=0.0)
                #We can safelly break here as there is only one nonbonded force in the system
                break
        
        #Double check that we are still internally consistent
        if fictional_particle_index == test == self.fictional_particle_index:
            pass
        else:
            raise ValueError("something went wrong when adding the fictional particle to the system")

    def get_field_trajectory(self, reset_context = True):
        """
        Runs a default workflow for calculating interaction energies on a trajectory.

        BEWARE: In this method no attempt is made to avoid the fictional particle interacting with itself
        over the PBC walls. The errors are on the order of 1e-3 kJ/(mol*debye) which is negligible compared
        to the fact that most enzymes have electric fields on the order of 20 kJ/(mol*debye).

        If you care about this, generate a comparison by reset_context(0,0,0,0) and
        then run get_field_trajectory(reset_context=False) to quantify the error per frame.
        Then remove that from whatever values you computed before.

        Parameters:
            reset_context : bool
                If False this does not alter the context and can hence be used as a utils function where
                the mother function alters context and this function returns the corresponding field
        
        Retruns:
            1d np.array of shape (traj frames) which contains the efield value on each frame
            The value is the projected length along efield_vec_sel[0] to efield_vec_sel[1] vector
            (probably the bond you selected). A positive value means that a +1 charge would want to
            move from efield_vec_sel[0] to efield_vec_sel[1]. 
            Unit is kJ/(mol*debye)
        """
        if reset_context:
            self.reset_context()

        efield = np.empty((len(self.u.trajectory)), dtype=np.float64)

        for frame_idx, ts in enumerate(self.u.trajectory):
            positions_np = ts.positions/10 #/10 to go from Å > nm
            self.context.setPositions(positions_np * unit.nanometer)

            #Get nonbonded forces from forcegroup 1 (if not asNumpy its a list of vec3)
            fictional_force = self.context.getState(forces=True, groups={1}).getForces(asNumpy=True)[self.fictional_particle_index].value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            
            #Project onto reference vector using the dot product
            fictional_force = np.dot(self.efield_vec_direction[frame_idx,:], fictional_force)
            efield[frame_idx] =  fictional_force

        #1D array of field strengths in kJ/(nm*mol)
        efield *= 0.02081943332291347

        return efield


    def get_field_decomposed(self, raw = False, progress_bar = False):
        """
        Runs a default workflow for calculating decomposed field.
        This is done by calculating a reference electric field over the trajectory and 
        then recalculating the trajectory with one residue at a time disabled.
        We then take the average difference over the trajectory to decompose how much of the field was from each residue

        Beware, this scales Nframes * Nresidues and can be quite expensive.

        Parameters:
            raw : bool
                whether you want the raw data matrix of shape (residue_idx+1, frame_idx) containing the dfield of a residue (+all solvent on last row) 
                in a certain trajectory frame. 
                or
                A pd.Dataframe "resid_idx", "resname", "dfield_avg", "dfield_std"
            progress_bar : bool
                Do you want a progress bar to track calculation progress?
        
        Returns:
            if raw is False
            Pd.Dataframe
                headers: "resid_idx", "resname", "dfield_avg", "dfield_std"
                dfield values in kJ/(mol*debye) for each residue.

            if raw is true
            Pd.Dataframe
                containing the electric field contribution in kJ/(mol*debye) for each residue and lastlly solvent over the trajectory
                so shape is (residue_idx+1, frame_idx).
        """
        #It would ofcourse be much more elegant to loop over residues calling self.get_field_trajectory
        #However since reading traj and setting positions is the majority of runtime this is approx 10x faster

        #Get a reference field trajectory with everything turned on and a freshlly cleaned context
        ref_efield = self.get_field_trajectory(reset_context=True)

        #get unique protein residue indexes
        protein_residue_idx = np.unique(self._protein_and_interesting_array[:,1])

        #store row for each residue (+1 for solvent) containing dfield over traj
        efield_traj = np.empty((len(protein_residue_idx) +1, len(self.u.trajectory)), dtype=np.float64)

        #Loop over protein
        for frame_idx, ts in tqdm(enumerate(self.u.trajectory), disable= not progress_bar, 
                                  desc="electric field decomposition", total=len(self.u.trajectory)):
            #Set positions to frame
            positions_np = ts.positions/10 #/10 to go from Å > nm
            self.context.setPositions(positions_np * unit.nanometer)

            #For residues in protein
            for resid_idx in protein_residue_idx:
                #Turn off our residue of interest and evaluate field
                self.context.setParameter(f"residue{resid_idx}_coul_scale", 0)

                #Get nonbonded forces from forcegroup 1 (if not asNumpy its a list of vec3)
                fictional_force = self.context.getState(forces=True, groups={1}).getForces(asNumpy=True)[self.fictional_particle_index].value_in_unit(unit.kilojoule_per_mole/unit.nanometer)

                #Project onto reference vector using the dot product
                fictional_force = np.dot(self.efield_vec_direction[frame_idx,:], fictional_force)

                #save into the output array
                efield_traj[resid_idx, frame_idx] = fictional_force

                #Turn on our residue of interest to reset the context
                self.context.setParameter(f"residue{resid_idx}_coul_scale", 1)

            #Calculate the solvent contribution
            self.context.setParameter(f"solvent_coul_scale", 0)

            #Get nonbonded forces from forcegroup 1 (if not asNumpy its a list of vec3)
            fictional_force = self.context.getState(forces=True, groups={1}).getForces(asNumpy=True)[self.fictional_particle_index].value_in_unit(unit.kilojoule_per_mole/unit.nanometer)

            #Project onto reference vector using the dot product
            fictional_force = np.dot(self.efield_vec_direction[frame_idx,:], fictional_force)

            #save into the output array
            efield_traj[-1, frame_idx] = fictional_force

            #Reset the context
            self.context.setParameter(f"solvent_coul_scale", 1)
        
        #postprocess output by converting kJ/(mol*nm) to kJ/(mol*debye)
        efield_traj *= 0.02081943332291347

        #Remove ref field and change sign to get field of ADDING a residue to the field
        efield_traj -= ref_efield
        efield_traj *= -1

        #Read out what residues names we ran analysis for so the user understands what is up, also add in solvent
        resnames = np.concatenate([self.u.residues.resnames[protein_residue_idx], np.array(["SOL"], dtype="O")], dtype="O")
        #Add solvent to protein residue_idx
        protein_residue_idx = np.concatenate([protein_residue_idx, np.array([np.max(protein_residue_idx)+1])])
        
        if raw == True:
            #retrieve axis labels for pd.dataframe
            frame_numbers = [frame for frame in range(len(self.u.trajectory))]
            residue_labels = []

            for index in range(len(protein_residue_idx)):
                residue_labels.append(resnames[index] + f"{protein_residue_idx[index]+1}")

            return pd.DataFrame(efield_traj, index=residue_labels, columns=frame_numbers)
        
        elif raw == False:
            #convert to the resid_idx, defield_avg, dfield_std array we desire
            efield_decomped = pd.DataFrame({"resid_idx": protein_residue_idx + 1,
                                            "resname" : resnames, 
                                            "dfield_avg" : np.average(efield_traj, axis=1),
                                            "dfield_std" : np.std(efield_traj, axis=1),
                                            })
            return efield_decomped

class Efield_amoeba(_Efield_base):

    def __init__(self,  topology, trajectory, scan_resid_sel, efield_vec_sel, polarization_type = "Mutual", 
                 platform="CUDA", precision = "single", solvent_names=("HOH", "CL", "NA"), 
                 sparsity=1, extra_decomp_residue_sel=["name MG", "name ZN"]):
        """
        Takes a finished simulation and does GPU accellerated electrostatic field calculations.
        This is still quite slow compared to normal MD if a polarizable force field is used.

        Internally this class uses OpenMM indexing which is 0 indexed for both atoms and residues.
        Usage with Mdanalysis uses normal MDA indexing which is 0 indexed for atoms and 1 indexed for residues.
        Hence one should never change internal variables after class creation to avoid confusion.

        If the scanned residue is part of the protein the code does run but it does not remove the scanned
        residues contribution to the field which will be quite large

        This class works with electric fields in kJ/(mol*debye)
        To convert to the experimental unit MV/cm multiply by 4.978170852464958

        For a reference workload the relative runtimes traj R^2 and decomp R^2 (amoeba mutual as reference) were
            Efield_amber                            3%, 0.60, 0.72
            Efield_amoeba polarization Direct       59%, 0.74, 0.28
            Efield_amoeba polarization Extrapolated 79%, 0.996, 0.999
            Efield_amoeba polarization Mutual       100%, 1, 1

        Parameters:
            topology : str or OpenMM.Topology
                Path to pdb file or OpenMM.Topology object.
            trajectory : str
                Path to trajectory file, e.g. dcd.
            scan_resid_sel : str
                MDAnalysis selection string for the residue in which you want the efield calculation
                to happen.
            efield_vec_sel : tuple of str
                2 member tuple of selections yielding 2 atoms who define the approximation to the transition
                state dipole moment. The fictional particle will be added halfway between these.
            polarization_type : str
                What level of polarization you want to run amoeba at.
                Valid choices are "direct", "extrapolated", "mutual"
                listed in order of increasing accuracy and cost, not case sensitive
            platform : str
                The OPENMM platform you want to run this class in.
                Valid choices are "CUDA", "HIP", "OpenCL" or "CPU"
                case sensitive
            precision : str
                What precision to evaluate forces on.
                valid choices are "single" and "double"
                Choosing single gives a 15x speedup but leads to a small random
                error on the fifth decimal and below
            solvent_names : tuple of str
                The residue names which are characterized as solvent.
            sparsity : int
                How often do you want to analyze the trajectory? 1=every frame
            extra_decomp_residue_sel : list of mdanalysis selection strings
                Extra selections for residues/atoms which should enter the decomposition.
                These residues are not included in the get_pointcharge_field() method.

        """

        #we have to rewrite init here to set polarization type
        dict_map = {"MUTUAL":AmoebaMultipoleForce.Mutual, "EXTRAPOLATED":AmoebaMultipoleForce.Extrapolated,
                    "DIRECT":AmoebaMultipoleForce.Direct}
        self.polarization_type = dict_map[polarization_type.upper()]

        super().__init__(topology=topology, trajectory=trajectory, scan_resid_sel=scan_resid_sel, 
                         efield_vec_sel=efield_vec_sel,  platform=platform, precision=precision, 
                         solvent_names=solvent_names, extra_decomp_residue_sel=extra_decomp_residue_sel, sparsity=sparsity)

    def _run_setup(self):
        """
        Runs internal setup specific to amber ff
        """

        self._biomolecule_ff = ["amoeba2018.xml"]
        self._create_system()
        

    def _init_params(self):
        """
        Initialize parameters and modify NonbondedForce for energy decomposition.
        Run in create_system()

        Forcegroups
        0 = not intresting
        1 = AmoebaMultipoleForce
        """
        #Iterate over all forces and divide into forcegroups

        for force in self.system.getForces():
            #Find the one NB force
            if isinstance(force, AmoebaMultipoleForce):
                self._amoebamultipoleforce = force
                force.setPolarizationType(self.polarization_type)

                #Assign nonbonded forcegroup
                force.setForceGroup(1)
                #We have to work differentlly from amber since Amoeba does not have addParticleParameterOffset
                #Workaround is to update particle parameters in context

                #Save the default parameters for each particle
                #list of lists (charge, dipole, quadropole, polarity)
                self._atom_params = []

                for atom_idx in range(force.getNumMultipoles()):
                    params = force.getMultipoleParameters(atom_idx)
                    self._atom_params.append(params)
                
                self._atom_params = pd.DataFrame(self._atom_params, columns=["charge", "dipole", "quadropole",  "axistype", "atomz", "atomx", "atomy", "thole", "damping", "polarity"])

            else:
                #Set not nonbonded forces to force group 0
                force.setForceGroup(0)
    
    def _add_fictional_particle_parameters(self):
        """ 
        Adds the fictional particle parameters to the system.
        Needs to be implemented for each force field as the nonbonded handling differs between force fields.

        Parameters:
            None

        """

        #Add the fictional massless particle to the system 
        fictional_particle_index = self.system.addParticle(0.0)  
        

        #Find nonbonded force
        for f in self.system.getForces():
            if isinstance(f, AmoebaMultipoleForce):
                test = f.addMultipole(1, molecularDipole = [0,0,0], molecularQuadrupole = [0,0,0, 0,0,0, 0,0,0], axisType=AmoebaMultipoleForce.NoAxisType, 
                               multipoleAtomZ=-1, multipoleAtomX=-1, multipoleAtomY=-1, thole=0, dampingFactor=0, polarity=0)

            elif isinstance(f, AmoebaVdwForce):
                #We do a if-else since openmm apparentlly includes 2 different apis for adding amoeba particles
                #We use sigma =1 epsilon = 0 since sigma=0 causes divide by 0 errors and epsilon=0 is good enough to cause
                #The vdw to always evaluate to 0
                if f.getUseParticleTypes():
                    dummy_type = f.addParticleType(1,0)
                    f.addParticle(fictional_particle_index, dummy_type, 0)

                else:
                    f.addParticle(fictional_particle_index, 1, 0, 0)

        #Double check that we are still internally consistent
        if fictional_particle_index == test == self.fictional_particle_index:
            pass
        else:
            raise ValueError("something went wrong when adding the fictional particle to the system")
    
    def reset_context(self, prot_coul=1, solv_coul=1):
        """
        Reset context parameters to defaults.
        Coloumb interactions are set to a value of 1 aka all forces are on except 
        the ones pertaining to the residue the fictional particle is inside which are 0.

        Parameters:
            prot_coul : float
                Scaling factor for protein coulombic parameters (default 1)
            solv_coul : float
                Scaling factor for solvent coulombic parameters (default 1)

        Returns:
            None
        """
        #We do this if we later need to refactor the code to handle induced and permanent dipoles differentlly
        #charge, multipole, polarization
        #Providing multipole scales as a int causes a silent python interpreter crash so we convert to float
        protein_scales =  [float(prot_coul), float(prot_coul), float(prot_coul)]
        solvent_scales = [float(solv_coul), float(solv_coul), float(solv_coul)]

        for atom_index in self.u.atoms.ix:
            #Check if atom is protein
            if atom_index in self._protein_and_interesting_array[:,0]:
                scales = protein_scales

            #If atom is solvent, no need to slice as we only store atom indexes
            elif atom_index in self._solvent_array:
                scales = solvent_scales
            
            #Add back original parameters to the fictional particle since it is perfect
            elif atom_index == self.fictional_particle_index:
                scales = [1,1,1]
            
            #Actually change the parameters
            self._amoebamultipoleforce.setMultipoleParameters(atom_index, 
                                                                  self._atom_params["charge"][atom_index] * scales[0], 
                                                                  self._atom_params["dipole"][atom_index] * scales[1], 
                                                                  self._atom_params["quadropole"][atom_index] * scales[1],
                                                                  self._atom_params["axistype"][atom_index],
                                                                  self._atom_params["atomz"][atom_index],
                                                                  self._atom_params["atomx"][atom_index],
                                                                  self._atom_params["atomy"][atom_index],
                                                                  self._atom_params["thole"][atom_index],
                                                                  self._atom_params["damping"][atom_index],
                                                                  self._atom_params["polarity"][atom_index] * scales[2])

        #update the context to reflext the new forces
        self._amoebamultipoleforce.updateParametersInContext(self.context)

    def _set_resid_params(self, resid ,coul_param):
        """ 
        Changes the multipole parameter of all atoms in a residue.
        We use this function since OpenMM does not have globalparameters for
        amoebas nonbonded forcetypes

        Parameters:
            resid : int
                The zero indexed residue index which we should change.
                -1 has the special meaning of the solvent atomgroup
            coul_param : float
                Scaling factor for resid coulombic parameters
        """

        #We do this if we later need to refactor the code to handle induced and permanent dipoles differentlly
        #charge, multipole, polarization
        #Providing multipole scales as a int causes a silent python interpreter crash so we convert to float

        #it might be physical to hold polarization at =1 here as there will always be something in a position which can be polarized
        scales =  [float(coul_param), float(coul_param), float(coul_param)]

        #handle the solvent case
        if resid == -1:
            atom_idxs = self._solvent_array
        
        #Handle nonspecial residue indexes
        else:
            atom_idxs = np.where(resid == self._protein_and_interesting_array[:,1])[0]
        
        for atom_index in atom_idxs:
            self._amoebamultipoleforce.setMultipoleParameters(atom_index, 
                                                                  self._atom_params["charge"][atom_index] * scales[0], 
                                                                  self._atom_params["dipole"][atom_index] * scales[1], 
                                                                  self._atom_params["quadropole"][atom_index] * scales[1],
                                                                  self._atom_params["axistype"][atom_index],
                                                                  self._atom_params["atomz"][atom_index],
                                                                  self._atom_params["atomx"][atom_index],
                                                                  self._atom_params["atomy"][atom_index],
                                                                  self._atom_params["thole"][atom_index],
                                                                  self._atom_params["damping"][atom_index],
                                                                  self._atom_params["polarity"][atom_index] * scales[2])

        self._amoebamultipoleforce.updateParametersInContext(self.context)




    def get_field_trajectory(self, reset_context = True):
        """
        Runs a default workflow for calculating interaction energies on a trajectory.

        BEWARE: In this method no attempt is made to avoid the fictional particle interacting with itself
        over the PBC walls. The errors are on the order of 1e-3 kJ/(mol*debye) which is negligible compared
        to the fact that most enzymes have electric fields on the order of 20 kJ/(mol*debye).

        If you care about this, generate a comparison by reset_context(0,0,0,0) and
        then run get_field_trajectory(reset_context=False) to quantify the error per frame.
        Then remove that from whatever values you computed before.

        Parameters:
            reset_context : bool
                If False this does not alter the context and can hence be used as a utils function where
                the mother function alters context and this function returns the corresponding field
        
        Retruns:
            1d np.array of shape (traj frames) which contains the efield value on each frame
            The value is the projected length along efield_vec_sel[0] to efield_vec_sel[1] vector
            (probably the bond you selected). A positive value means that a +1 charge would want to
            move from efield_vec_sel[0] to efield_vec_sel[1]. 
            Unit is kJ/(mol*debye)
        """
        if reset_context:
            self.reset_context(1,1)

        efield = np.empty((len(self.u.trajectory)), dtype=np.float64)

        for frame_idx, ts in enumerate(self.u.trajectory):
            positions_np = ts.positions/10 #/10 to go from Å > nm
            self.context.setPositions(positions_np * unit.nanometer)

            #Get nonbonded forces from forcegroup 1 (if not asNumpy its a list of vec3)
            fictional_force = self.context.getState(forces=True, groups={1}).getForces(asNumpy=True)[self.fictional_particle_index].value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            
            #Project onto reference vector using the dot product
            fictional_force = np.dot(self.efield_vec_direction[frame_idx,:], fictional_force)
            efield[frame_idx] =  fictional_force

        #1D array of field strengths in kJ/(nm*mol)
        efield *= 0.02081943332291347

        return efield
    
    def get_field_decomposed(self, raw = False, progress_bar=True):
        """
        Runs a default workflow for calculating decomposed field.
        This is done by calculating a reference electric field over the trajectory and 
        then recalculating the trajectory with one residue at a time disabled.
        We then take the average difference over the trajectory to decompose how much of the field was from each residue

        Beware, this scales Nframes * Nresidues and can be quite expensive.

        Parameters:
            raw : bool
                whether you want the raw data matrix of shape (residue_idx+1, frame_idx) containing the dfield of a residue (+all solvent on last row) 
                in a certain trajectory frame. 
                or
                A pd.Dataframe "resid_idx", "resname", "dfield_avg", "dfield_std"
            progress_bar : bool
                Do you want a progress bar to track calculation progress?
        
        Returns:
            if raw is False
            Pd.Dataframe
                headers: "resid_idx", "resname", "dfield_avg", "dfield_std"
                dfield values in kJ/(mol*debye) for each residue.

            if raw is true
            Pd.Dataframe
                containing the electric field contribution in kJ/(mol*debye) for each residue and lastlly solvent over the trajectory
                so shape is (residue_idx+1, frame_idx).
        """

        #Get a reference field trajectory with everything turned on and a freshlly cleaned context
        ref_efield = self.get_field_trajectory(reset_context=True)

        #get unique protein residue indexes
        protein_residue_idx = np.unique(self._protein_and_interesting_array[:,1])

        #store row for each residue (+1 for solvent) containing dfield over traj
        efield_traj = np.empty((len(protein_residue_idx) +1, len(self.u.trajectory)), dtype=np.float64)

        #Loop over protein
        for frame_idx, ts in tqdm(enumerate(self.u.trajectory), disable= not progress_bar, 
                                  desc="electric field decomposition", total=len(self.u.trajectory)):
            #Set positions to frame
            positions_np = ts.positions/10 #/10 to go from Å > nm
            self.context.setPositions(positions_np * unit.nanometer)

            #For residues in protein
            for resid_idx in protein_residue_idx:
                #Turn off our residue of interest and evaluate field
                self._set_resid_params(resid_idx, 0)

                #Get nonbonded forces from forcegroup 1 (if not asNumpy its a list of vec3)
                fictional_force = self.context.getState(forces=True, groups={1}).getForces(asNumpy=True)[self.fictional_particle_index].value_in_unit(unit.kilojoule_per_mole/unit.nanometer)

                #Project onto reference vector using the dot product
                fictional_force = np.dot(self.efield_vec_direction[frame_idx,:], fictional_force)

                #save into the output array
                efield_traj[resid_idx, frame_idx] = fictional_force

                #Turn on our residue of interest to reset the context
                self._set_resid_params(resid_idx, 1)

            #Calculate the solvent contribution
            self._set_resid_params(-1, 0)
            #Get nonbonded forces from forcegroup 1 (if not asNumpy its a list of vec3)
            fictional_force = self.context.getState(forces=True, groups={1}).getForces(asNumpy=True)[self.fictional_particle_index].value_in_unit(unit.kilojoule_per_mole/unit.nanometer)

            #Project onto reference vector using the dot product
            fictional_force = np.dot(self.efield_vec_direction[frame_idx,:], fictional_force)

            #save into the output array
            efield_traj[-1, frame_idx] = fictional_force

            #Reset the context
            self._set_resid_params(-1, 1)
        
        #postprocess output by converting kJ/(mol*nm) to kJ/(mol*debye)
        efield_traj *= 0.02081943332291347

        #Remove ref field and change sign to get field of ADDING a residue to the field
        efield_traj -= ref_efield
        efield_traj *= -1

        #Read out what residues names we ran analysis for so the user understands what is up, also add in solvent
        resnames = np.concatenate([self.u.residues.resnames[protein_residue_idx], np.array(["SOL"], dtype="O")], dtype="O")
        #Add solvent to protein residue_idx
        protein_residue_idx = np.concatenate([protein_residue_idx, np.array([np.max(protein_residue_idx)+1])])
        
        if raw == True:
            #retrieve axis labels for pd.dataframe
            frame_numbers = [frame for frame in range(len(self.u.trajectory))]
            residue_labels = []

            for index in range(len(protein_residue_idx)):
                residue_labels.append(resnames[index] + f"{protein_residue_idx[index]+1}")

            return pd.DataFrame(efield_traj, index=residue_labels, columns=frame_numbers)
        
        elif raw == False:
            #convert to the resid_idx, defield_avg, dfield_std array we desire
            efield_decomped = pd.DataFrame({"resid_idx": protein_residue_idx + 1,
                                            "resname" : resnames, 
                                            "dfield_avg" : np.average(efield_traj, axis=1),
                                            "dfield_std" : np.std(efield_traj, axis=1),
                                            })
            return efield_decomped
        

def main():
    """ 
    Commandline parser for running just the efield analysis
    """

    #Start the command line parser
    parser = argparse.ArgumentParser(
                        prog="whatcat-efield",
                        description=(
                        """ 
                        This script runs a electrostatic field analysis on a given trajectory and topology.
                        """))

    parser.add_argument("topology", type = str, help = "PDB structure of the structure you want to simulate (pdb or openmm system xml).") 
    parser.add_argument("trajectory", type=str, help="The trajectory you want to analyze")

    parser.add_argument("-rt", type=int, help="What was the reporting time of the trajectory in ps?", required=True)
    parser.add_argument("-sparse", "--sparsity", default=1, type=int, help="how sparse do you want the trajectory to be?")
    parser.add_argument("--platform", type = str, default= "CUDA", help="""Sets the simulation platform and optionally device if specified as a comma separated string of ints eg "CUDA_0" or "CUDA_0,1", default = "CUDA" """, required=False)
    
    parser.add_argument("-efield_vec","--efield_vec", type = str, help="""a list of 3 of atom selectors returning [efield_residue, efield_vec_positive, efield_vec_negative]. 
                        specify using MDAnalysis/VMD natural language queries""", required=True)
    parser.add_argument("-efield_decomp","--efield_decomp", type = str, default= "false", choices=["true", "True", "false", "False"], help="Whether to run efield decomposition and pointcharge field calculation", required=False)
    parser.add_argument("-efield_method","--efield_method", type = str, choices=["amber", "amoeba_mutual", "amoeba_extrapolated", "amoeba_direct"], default= "amber", help="What forcefield to use for electric field decomposition", required=False)
    parser.add_argument("-decomp_raw", type=str, default="False", choices=["true", "True", "false", "False"],help="Whether you want the decomposition values reported every single frame. This is useful if analyzing cluster medoids", required=False)
    # Parse arguments
    args = parser.parse_args()

    start_time = time.time()
    print("Starting efield analysis")

    method = args.efield_method
    trajectory = args.trajectory
    raw = utils.str_to_bool(args.decomp_raw)

    if method.upper() == "AMBER":
        efield_calc = Efield_amber(args.topology, args.trajectory, utils.css_to_list(args.efield_vec)[0], 
                                       utils.css_to_list(args.efield_vec)[1:3], args.platform, sparsity=args.sparsity)
        
    elif method.upper().split("_")[0] == "AMOEBA":
        efield_calc = Efield_amoeba(args.topology, args.trajectory, utils.css_to_list(args.efield_vec)[0], 
                                    utils.css_to_list(args.efield_vec)[1:3], 
                                    method.upper().split("_")[1], args.platform, sparsity=args.sparsity)
    else:
        raise ValueError("invalid method input")
    
    efield_traj = efield_calc.get_field_trajectory()
    frame_times = [index * args.rt * args.sparsity for index in range(len(efield_calc.u.trajectory))] 
    pd.DataFrame({"Time (ps)":frame_times, f"efield_{utils.css_to_list(args.efield_vec)[0].replace(" ", "_")}":efield_traj}).to_csv("efield_traj.csv")
    
    if utils.str_to_bool(args.efield_decomp):
        residue_df = efield_calc.get_field_decomposed(raw=raw)
        efield_pointcharge = efield_calc.get_pointcharge_field(raw=raw)
        

        if raw==False:
            residue_df = pd.concat([residue_df, pd.DataFrame({"pointcharge_field":efield_pointcharge})], axis=1)
            residue_df.to_csv("efield_decomp.csv", index=False)
        
        elif raw==True:
            residue_df.to_csv("efield_decomp.csv", index=True)

            #Add in residue information to pointcharge array
            resnames = efield_calc.u.atoms.select_atoms("protein").residues.resnames + (efield_calc.u.atoms.select_atoms("protein").residues.ix_array+1).astype(str)
            pd.DataFrame(efield_pointcharge.T, index = resnames).to_csv("efield_decomp_pointcharge.csv", index=True)


    print(f"{round(time.time() - start_time,2)}s used for electric fields")


if __name__ == "__main__":
    main()




