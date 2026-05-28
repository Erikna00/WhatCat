import numpy as np
import MDAnalysis as mda 
from MDAnalysis.analysis.distances import dist
from MDAnalysis.lib.distances import calc_angles, calc_dihedrals
import mdaencore
import sklearn.cluster

def compute_geometric_params_frame(atomgroup, atom_selector_lists):
    """
    Compute geometric parameters for all specified atom selector lists on the current frame.
    Used as a helper function in whatcat_md_analysis.compute_geometric_params.
    
    Parameters
        atomgroup : MDAnalysis.core.groups.AtomGroup
            The atom group updated for the current frame.
        atom_selection_list : list of list(str, str)
            List of list, each sublist containing 2-4 MDAnalysis atom selector strings.
    
    Returns
        np.ndarray
            1D array of distances for each specified atom pair for the current frame.
    """

    #generate list of lists for data storage
    geometric_params = []
    for selection_list in atom_selector_lists:
        
        selected_atoms = [atomgroup.select_atoms(sel) for sel in selection_list]

        #check that all selections were valid
        for selection in selected_atoms:
            if len(selection) != 1:
                raise ValueError(f"Each selection must match exactly one atom: currentlly \n{selection_list} \ngave \n{selected_atoms}")
        
        #calculate distance
        if len(selected_atoms) == 2:
            # Use MDAnalysis's 'dist' function; it returns an array, and the last element is the
            # distance between the two atoms. (the first two are residue ID:s)
            d = dist(selected_atoms[0], selected_atoms[1])[-1, 0]
            geometric_params.append(d)

        #calculate angle
        if len(selected_atoms) == 3:
            angles_rad = calc_angles(selected_atoms[0].positions, selected_atoms[1].positions, 
                                     selected_atoms[2].positions)
            angles_deg = np.degrees(angles_rad)
            angles_deg = float(angles_deg)

            geometric_params.append(angles_deg)

        #calculate dihedral
        if len(selected_atoms) == 4:
            angles_rad = calc_dihedrals(selected_atoms[0].positions, selected_atoms[1].positions, 
                                     selected_atoms[2].positions, selected_atoms[3].positions)
            angles_deg = np.degrees(angles_rad)
            angles_deg = float(angles_deg)
            geometric_params.append(angles_deg)
    
    return np.array(geometric_params)

def compute_rmsf_chunk(pdb_filename, traj_filename, frame_indices, selection, ref_pdb):
    """
    Compute mean squared fluctuations for a subset of frames, applying on‐the‐fly alignment to ref_pdb.
    
    Parameters:
        pdb_filename : str 
            Path to topology file.
        traj_filename : str
            Path to trajectory file.
        frame_indices : list of int 
            Frames assigned to this worker.
        selection : str
            Atom selection string.
        ref_positions : np.ndarray 
            Reference positions (for the selected atoms) for alignment.
        
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
        pdb_filename : str 
            Path to the topology file.
        traj_filename : str 
            Path to the trajectory file.
        selection : str 
            Atom selection string for RMSD calculation.
        frame_pairs : list of tuples
            Pairs of frame indices (i, j) to compute RMSD.

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
    """
    This function calculates radius of gyration for one frame in a MDA trajectory.
    Used by whatcat_md_analysis.calc_rgyr.
    """

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


def covariance_to_correlation(covariance_matrix):
    """
    Convert a covariance matrix from mdanalysis to a correlation matrix.
    
    Parameters:
        covariance_matrix : np.ndarray
            The covariance matrix to convert.
    
    Returns:
        np.ndarray: The resulting correlation matrix.
    """
    diag = np.sqrt(np.diag(covariance_matrix))
    outer_diag = np.outer(diag, diag)
    correlation_matrix = covariance_matrix / outer_diag
    np.fill_diagonal(correlation_matrix, 1.0)  # Set diagonal to 1
    
    return correlation_matrix

def matrix_3M_to_M(matrix_3M):
    """
    Convert a 3M x 3M matrix to an M x M matrix by averaging over 3x3 blocks.
    This is useful for reducing covariance or correlation matrices of atomic positions x,y,z
    to a distance covariance or correlation matrix per atom.
    
    Parameters:
        matrix_3M : np.ndarray
            The input 3M x 3M matrix.
    
    Returns:
        np.ndarray: The resulting M x M matrix.
    """
    if matrix_3M.shape[0] % 3 != 0 or matrix_3M.shape[1] % 3 != 0:
        raise ValueError("Input matrix dimensions must be multiples of 3.")
    
    size = matrix_3M.shape[0] // 3
    matrix_M = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            block = matrix_3M[i*3:(i+1)*3, j*3:(j+1)*3]
            matrix_M[i, j] = np.mean(block)
    
    return matrix_M


def list_contigous_range(list_of_ints):
    """
    Helper function to find contigous ranges in a list of integers.
    Used in whatcat_md.md.MDCluster.cluster_analysis for finding contigous frames in clusters.

    Parameters:
        list_of_ints : list of int
            List of integers to analyze.

    Returns:
        list of tuples : Each tuple contains (min, max) of a contigous range

    """
    
    min_max_list = []

    # sort to ensure contiguous ranges are found correctly
    sorted_ints = sorted(list_of_ints)

    range_start = None
    last_element = None

    for element in sorted_ints:
        if range_start is None:
            range_start = element
            last_element = element
        elif element == last_element + 1:
            last_element = element
        else:
            min_max_list.append((range_start, last_element))
            range_start = element
            last_element = element

    # append the final open range
    min_max_list.append((range_start, last_element))

    return min_max_list
        


class HDBscan_mdaencore(mdaencore.clustering.ClusteringMethod.ClusteringMethod):
    def __init__(self, min_cluster_size=5, **kwargs):
        """
        Interface to the HDBscan propagation clustering procedure implemented
        in sklearn. This class can be used as a clustering method in mdaencore using some trickery
        to access medoids since sklearns HDBSCAN does not provide medoids when using precomputed distance matrices.

        Internally we make use of sklearn.cluster.HDBSCAN.
        Then we assign all noise to the nearest cluster medoid to make sure all frames are assigned to a cluster
        since mdaencore clustering framework does not handle noise points. This is preferable to mdaencores DBSCAN
        since that one just picks a centroid at random from the core points and assigns all noise to its own cluster.

        Parameters
            min_cluster_size : int
                Parameter for the HDBSCAN clustering.

            **kwargs : optional
                Other keyword arguments are passed to :class:`sklearn.cluster.HDBSCAN`.

        """
        self.hdbscan = sklearn.cluster.HDBSCAN(min_cluster_size=min_cluster_size, metric="precomputed", **kwargs)

    def __call__(self, distance_matrix):
        """
        Parameters
        ----------

        distance_matrix : encore.utils.TriangularMatrix
            conformational distance matrix

        Returns:
            numpy.array : array, shape(n_elements) 
                centroid frames of the clusters for all of the elements
        """

        # Fit HDBSCAN on the full (precomputed) distance array
        distance_matrix = distance_matrix.as_array()
        clusters = self.hdbscan.fit_predict(distance_matrix)

        if not np.any(clusters > -1):
            print("No cluster found :(, will now reassign all noise to cluster 0")
            clusters[:] = 0

        #sklearn cannot find medoids when using a precomputed distance matrix
        unique_clusters = [cluster for cluster in np.unique(clusters) if cluster >= 0]
        medoids = []

        for cluster in unique_clusters:
            idxs = np.where(clusters == cluster)[0]

            # pairwise distances inside cluster
            sub = distance_matrix[np.ix_(idxs, idxs)]
            
            # sum distances for each candidate medoid
            sums = np.sum(sub, axis=1)
            medoid_local = int(np.argmin(sums))
            medoid_global = int(idxs[medoid_local])
            medoids.append(medoid_global)

        # Attach medoids_ in the same order as unique_clusters
        if len(medoids) > 0:
            self.hdbscan.medoids_ = np.array(medoids, dtype=int)

        #Now we need to map frames assigned as noise (-1) to the nearest cluster/medoid
        #We do this since mdaencore expects all frames to be assigned to a cluster medoid and does not handle noise points
        noise_idx = np.where(clusters == -1)[0]
        d_noise_to_medoids = distance_matrix[noise_idx][:, medoids]  # shape (n_noise, n_medoids)
        nearest = np.argmin(d_noise_to_medoids, axis=1) #contains cluster id each noise point belongs to

        #actually reassign noise points to nearest medoid
        for i, noise_idx in enumerate(noise_idx):
            clusters[noise_idx] = nearest[i]

        #We need to change clusters to point at medoid frame instead of cluster label
        medoid_clusters = [medoids[entry] if entry >= 0 else entry for entry in clusters]

        # Encode centroid info for mdaencore (expects clusters and medoids_)
        clusters = mdaencore.clustering.ClusteringMethod.encode_centroid_info(medoid_clusters, self.hdbscan.medoids_)

        return clusters